# The two SNN configurations of the final pipeline, shared by every training
# script (sourced from the submission directory, which Slurm uses as the
# working directory):
#
#   pooled       run_snn_pooled_pretrain.sbatch + run_snn_pooled_finetune_array.sbatch:
#                the PyTorch SNN ('snn' in inference). 512 -> 256 -> 128, threshold
#                1.0, 2 EMA stages, tau_syn initialized at 2, 4, 8, 16 over the four
#                spiking layers (trained). Pretrained with hard reset
#                (select_model pooled_pretrain), fine-tuned with soft reset (select_model
#                pooled), the most accurate in PyTorch; the reset has no weights, so the
#                hard-reset pretrained weights load unchanged. Not deployable to Speck
#                (no synaptic stage).
#   lsnn         The DEFAULT of the two pooled scripts (MODEL=lsnn): the recurrent LSNN
#                (train_snn.py --arch lsnn, models/model_bmi_lsnn.py) through the
#                same LOSO pretrain -> fine-tune pipeline: 2 x 256 recurrent IAF,
#                90% adaptive, synaptic filters 20 ms-1.3 s, readout from both layers,
#                two readout stages. Its checkpoints go to lsnn_* directories; the
#                pooled SNN above stays available with MODEL=pooled (its own
#                directories, untouched).
#   per_session  run_snn_per_session_array.sbatch: the Speck SNN ('speck'). 256 -> 128
#                -> 64, threshold 1.0, 2 EMA stages, no tau_syn, hard reset (soft
#                reset loses more accuracy on the chip). hkm trains the same way; its
#                dense input is handled at deployment (hidden layer 1's threshold x2
#                and a 3 ms wait per step on the chip, run_inference.sbatch).
#
# Every model trains on binarized input (train_snn.py --binarize-input, default).
#
# Requires DATA_ROOT, EXPERIMENT and SUBJECT; call select_model
# pooled_pretrain|pooled|per_session.
# LOSO (pooled only): 1 (default) pretrains one model per session on every OTHER
# session of the subject, so a session's own data never reaches its pretraining;
# 0 pretrains one model on every session.

CONFIG_YAML="../configs/iaf_hard_reset.yaml"
TRAIN_SCRIPT="../snn_training/train_snn.py"
POOL_SCRIPT="../preprocessing_training/build_pretraining_pool.py"

# bmi: grouped datasets, batches of 20 equal-length trials.
# hkm: whole-trial datasets of different lengths, one trial per batch.
case "$EXPERIMENT" in
    hkm) DATASET_SUBDIR="mua"; BATCH_SIZE=1 ;;
    *)   DATASET_SUBDIR="mua_8_group"; BATCH_SIZE=20 ;;
esac
SNN_DATASET_ROOT="${DATA_ROOT}/snn_datasets/${EXPERIMENT}/${SUBJECT}/${DATASET_SUBDIR}"
CHECKPOINT_BASE="${DATA_ROOT}/snn_checkpoints/${EXPERIMENT}/${SUBJECT}"

LOSO="${LOSO:-1}"
case "$LOSO" in
    0) POOL_KIND="full_cohort" ;;
    1) POOL_KIND="loso" ;;
    *) echo "ERROR: LOSO must be 0 or 1, got '$LOSO'" >&2; exit 1 ;;
esac
# Checkpoint directories end in _binarized (binarized input), and the
# fine-tuned ones in _binarized_soft (soft reset), so they never meet the
# checkpoints of earlier runs, which the skip/resume checks would otherwise take
# for this pipeline's.
FINETUNE_ROOT="${CHECKPOINT_BASE}/${POOL_KIND}_finetuned_binarized_soft"
PER_SESSION_ROOT="${CHECKPOINT_BASE}/per_session_binarized"

# apply_val_frac: with VAL_FRAC set (e.g. 0.1), train with a validation split for model selection (train_snn.py
# --val-frac) and write to checkpoint directories ending in _val, so the original (test-selected) runs are kept.
# Pretrained checkpoints (pretrain_dir) are read from their original directories: pretraining never sees the
# held-out session.
apply_val_frac() {
    [ -n "${VAL_FRAC:-}" ] || return 0
    TRAIN_ARGS+=(--val-frac "$VAL_FRAC")
    FINETUNE_ROOT="${FINETUNE_ROOT}_val"
    PER_SESSION_ROOT="${PER_SESSION_ROOT}_val"
    MODEL_DESC="${MODEL_DESC}, selected on the last ${VAL_FRAC} of the training trials"
}

# select_model pooled_pretrain|pooled|per_session -> the network and training
# controls (pooled_pretrain: the pooled network with hard reset).
# Epochs are maxima (early stopping); hkm epochs take about an hour per
# session, so hkm runs fewer and checkpoints every epoch.
select_model() {
    EXTRA_ARGS=()
    case "$1" in
        pooled|pooled_pretrain)
            HIDDEN_DIMS=(512 256 128); THRESHOLDS=(1.0 1.0 1.0 1.0); TAU_SYN=(2 4 8 16)
            RESET_TYPE=$([ "$1" = "pooled_pretrain" ] && echo hard || echo soft)
            if [ "$EXPERIMENT" = "hkm" ]; then
                EPOCHS=10; PATIENCE=5; SCHEDULER_PATIENCE=5; CHECKPOINT_INTERVAL=1
            else
                EPOCHS=50; PATIENCE=10; SCHEDULER_PATIENCE=10; CHECKPOINT_INTERVAL=10
            fi ;;
        per_session)
            HIDDEN_DIMS=(256 128 64); THRESHOLDS=(1.0 1.0 1.0 1.0); TAU_SYN=()
            RESET_TYPE=hard
            if [ "$EXPERIMENT" = "hkm" ]; then
                EPOCHS=20; PATIENCE=5; SCHEDULER_PATIENCE=5; CHECKPOINT_INTERVAL=1
            else
                EPOCHS=50; PATIENCE=10; SCHEDULER_PATIENCE=5; CHECKPOINT_INTERVAL=10
            fi ;;
        speck_tap1|speck_tap1_wide|speck_tap1_funnel)
            # no input delays (one tap per channel), for hkm's dense input: half the input events per step of
            # speck_delay2, to test whether the input load per step drives the chip's over-firing.
            #   speck_tap1       192 -> 128 -> 256 -> 256, speck_delay2's network without its delayed copy
            #   speck_tap1_wide  192 -> 256 -> 256 -> 128, the widest first layer that maps onto Speck2f
            #   speck_tap1_funnel 192 -> 256 -> 128 -> 64, the wide first layer narrowing to the output
            THRESHOLDS=(1.0 1.0 1.0 1.0); TAU_SYN=(); RESET_TYPE=hard
            case "$1" in speck_tap1) HIDDEN_DIMS=(128 256 256) ;; speck_tap1_funnel) HIDDEN_DIMS=(256 128 64) ;; *) HIDDEN_DIMS=(256 256 128) ;; esac
            PER_SESSION_ROOT="${CHECKPOINT_BASE}/per_session_${1#speck_}_binarized"
            if [ "$EXPERIMENT" = "hkm" ]; then
                EPOCHS=20; PATIENCE=5; SCHEDULER_PATIENCE=5; CHECKPOINT_INTERVAL=1
            else
                EPOCHS=50; PATIENCE=10; SCHEDULER_PATIENCE=5; CHECKPOINT_INTERVAL=10
            fi ;;
        speck_delay4)
            # speck_delay's delays (0, 40, 100, 200 ms) on the narrower network 384 -> 128 -> 128 -> 64 -> 36
            # (compare_speck_delays.py delay4): what the wider later layers add
            HIDDEN_DIMS=(128 128 64); THRESHOLDS=(1.0 1.0 1.0 1.0); TAU_SYN=()
            RESET_TYPE=hard
            EXTRA_ARGS=(--input-delays 0 10 25 50)
            PER_SESSION_ROOT="${CHECKPOINT_BASE}/per_session_delay4_binarized"
            if [ "$EXPERIMENT" = "hkm" ]; then
                echo "ERROR: speck_delay4 is bmi only" >&2; exit 1
            fi
            EPOCHS=50; PATIENCE=10; SCHEDULER_PATIENCE=5; CHECKPOINT_INTERVAL=10 ;;
        speck_delay|speck_delay2|speck_delay_pretrain|speck_delay_long|speck_delay_xlong|speck_delay_random|speck_delay_shift|speck_plan_prune5|speck_plan_tuned|speck_delay_learn|speck_delay_learn_shared)
            # the Speck SNN with the off-chip input delay bank (models/model_bmi.py delay_bank()): every
            # channel also sent with 3 delays, 384 chip inputs -> 128 -> 256 -> 256 -> 36, the largest that
            # maps onto Speck2f with 4 delays (snn_training/compare_speck_delays.py delay4_wide). Delays:
            #   speck_delay        0, 40, 100, 200 ms
            #   speck_delay2       0, 100 ms: hkm's 192 channels x 2 = 384 chip inputs (192 x 4 = 768 does not map)
            #   speck_delay_long   0, 100, 300, 600 ms
            #   speck_delay_xlong  0, 200, 600, 1200 ms
            #   speck_delay_random "now" plus 3 delays per channel, each channel's own, drawn from 4-200 ms
            #   speck_delay_shift  0, 80, 160, 240 ms (where the channels' velocity information lies)
            #   speck_plan_prune5  per session: only channels tuned to velocity, each at 0, 60, 120, 180, 240 ms
            #   speck_plan_tuned   per session: every channel "now" plus 3 delays around its own lead
            #   speck_delay_learn  0, 40, 100, 200 ms to start; the 3 delays of each channel trained within
            #                      0-300 ms (the cohort's window), then rounded and frozen after 15 epochs
            #   speck_delay_learn_shared  the same with 3 trained delays shared by all channels
            # (the plans: snn_training/channel_lag_tuning.py + make_input_plans.py, from each session's
            # training portion, in ${DATA_ROOT}/input_plans/<prune5|tuned>/<session>.json)
            # Trained per session like per_session, each into per_session_<name without speck_>_binarized.
            # MODEL=speck_delay on run_snn_pooled_pretrain.sbatch / run_snn_pooled_finetune_array.sbatch:
            # the same network pretrained leave-one-session-out (speck_delay_pretrain) and fine-tuned per
            # session, both hard reset (as on the chip), into speckdelay_<loso>_{pretrained,finetuned}_binarized
            HIDDEN_DIMS=(128 256 256); THRESHOLDS=(1.0 1.0 1.0 1.0); TAU_SYN=()
            RESET_TYPE=hard
            case "$1" in
                speck_delay2)       EXTRA_ARGS=(--input-delays 0 25) ;;
                speck_delay_long)   EXTRA_ARGS=(--input-delays 0 25 75 150) ;;
                speck_delay_xlong)  EXTRA_ARGS=(--input-delays 0 50 150 300) ;;
                speck_delay_random) EXTRA_ARGS=(--input-delays 0 10 25 50 --input-delays-random true) ;;
                speck_delay_shift)  EXTRA_ARGS=(--input-delays 0 20 40 60) ;;
                speck_plan_prune5)  EXTRA_ARGS=(--input-plan-dir "${DATA_ROOT}/input_plans/prune5") ;;
                speck_plan_tuned)   EXTRA_ARGS=(--input-plan-dir "${DATA_ROOT}/input_plans/tuned") ;;
                speck_delay_learn)  EXTRA_ARGS=(--input-delays 0 10 25 50 --learn-input-delays channel --input-delay-max 75) ;;
                speck_delay_learn_shared) EXTRA_ARGS=(--input-delays 0 10 25 50 --learn-input-delays shared --input-delay-max 75) ;;
                *)                  EXTRA_ARGS=(--input-delays 0 10 25 50) ;;
            esac
            PER_SESSION_ROOT="${CHECKPOINT_BASE}/per_session_${1#speck_}_binarized"
            if [ "$EXPERIMENT" = "hkm" ]; then     # model_hkm.py has the shared delay bank only; epochs as per_session
                case "$1" in speck_delay|speck_delay2|speck_delay_pretrain) ;; *) echo "ERROR: $1 is bmi only (hkm: speck_delay2; speck_delay's 768 inputs do not map onto Speck2f)" >&2; exit 1 ;; esac
                EPOCHS=20; PATIENCE=5; SCHEDULER_PATIENCE=5; CHECKPOINT_INTERVAL=1
            else
                EPOCHS=50; PATIENCE=10; SCHEDULER_PATIENCE=5; CHECKPOINT_INTERVAL=10
            fi ;;
        ladder_filters|ladder_filters_soft|ladder_rec|ladder_width|ladder_soft|ladder_chip)
            # The motor ladder (bmi, per session, as per_session): from the Speck SNN (per_session: feedforward
            # 96 -> 256 -> 128 -> 64 -> 36, hard reset, no synaptic filters), one change per step, each step
            # keeping the ones before:
            #   ladder_filters  + synaptic filters: tau_syn 2, 4, 8, 16 steps (8-64 ms), trained (PyTorch only)
            #   ladder_filters_soft  + soft reset, still feedforward (the LSNN's neurons need soft reset: as the
            #                   bridge, hard reset gave 47.9, soft 39.3, so soft reset comes first)
            #   ladder_rec      + recurrence in every hidden layer (--recurrent; PyTorch only)
            #   ladder_width    + width: 512 -> 256 -> 128
            #   ladder_soft     + soft reset
            #   (ladder_rec/width/soft are UNSTABLE: recurrence on this network runs away -- its sinabs filters
            #   amplify, spikes and membrane potentials are unbounded -- and every session ended at the zero-velocity
            #   fallback, 70.12. Superseded by lsnn_ladder_*, below.)
            # and the chip's branch from the same start:
            #   ladder_chip     the chip's shape for 384 inputs, 128 -> 256 -> 256, no delays yet; speck_delay
            #                   then adds the input delays (0, 40, 100, 200 ms)
            THRESHOLDS=(1.0 1.0 1.0 1.0); RESET_TYPE=hard; HIDDEN_DIMS=(256 128 64); TAU_SYN=()
            case "$1" in
                ladder_chip) HIDDEN_DIMS=(128 256 256) ;;
                ladder_filters_soft) TAU_SYN=(2 4 8 16); RESET_TYPE=soft ;;
                *)
                    TAU_SYN=(2 4 8 16)
                    [ "$1" != ladder_filters ] && EXTRA_ARGS=(--recurrent true)
                    case "$1" in ladder_width|ladder_soft) HIDDEN_DIMS=(512 256 128) ;; esac
                    [ "$1" = ladder_soft ] && RESET_TYPE=soft ;;
            esac
            PER_SESSION_ROOT="${CHECKPOINT_BASE}/per_session_${1}_binarized"
            if [ "$EXPERIMENT" = "hkm" ]; then
                echo "ERROR: $1 is bmi only" >&2; exit 1
            fi
            EPOCHS=50; PATIENCE=10; SCHEDULER_PATIENCE=5; CHECKPOINT_INTERVAL=10 ;;
        lsnn|lsnn_pretrain)
            # one network for both stages (no reset switch); epochs and patience as pooled
            HIDDEN_DIMS=(256 256); THRESHOLDS=(); TAU_SYN=(); RESET_TYPE=soft
            if [ "$EXPERIMENT" = "hkm" ]; then
                EPOCHS=10; PATIENCE=5; SCHEDULER_PATIENCE=5; CHECKPOINT_INTERVAL=1
            else
                EPOCHS=50; PATIENCE=10; SCHEDULER_PATIENCE=10; CHECKPOINT_INTERVAL=10
            fi ;;
        lsnn_ladder_ff|lsnn_ladder_soft|lsnn_ladder_rec|lsnn_ladder_width|lsnn_ladder_adapt)
            # The motor ladder's second half, on the LSNN's own code (models/model_bmi_lsnn.py), per session as
            # per_session. Recurrence added to the original SNN (ladder_rec) runs away: its sinabs filters amplify,
            # spikes and membrane potentials are unbounded, nothing adapts. So the ladder switches to the LSNN's
            # neuron model (at most 3 spikes per step, unit-gain per-neuron filters 20 ms-1.3 s, membrane ceiling,
            # leaky readout of every layer) at the original shape, then one change per step, each keeping the ones
            # before. Soft reset comes before recurrence: with hard reset, recurrence and width trained unstably
            # (test loss swinging up to 2-3x; local check, 6 epochs on indy_20160411_01), with soft reset steadily.
            #   lsnn_ladder_ff     the LSNN neuron model, feedforward 256 -> 128 -> 64, hard reset, no adaptation
            #   lsnn_ladder_soft   + soft reset
            #   lsnn_ladder_rec    + recurrence
            #   lsnn_ladder_width  + width: 2 x 256 (the final LSNN's shape, one layer fewer)
            #   lsnn_ladder_adapt  + adaptation in 90% of neurons = the final LSNN, trained per session
            # (the final LSNN itself adds leave-one-session-out pretraining: MODEL=lsnn)
            HIDDEN_DIMS=(256 128 64); THRESHOLDS=(); TAU_SYN=(); RESET_TYPE=soft
            case "$1" in
                lsnn_ladder_ff)    RESET_TYPE=hard; LSNN_FLAGS=(--lsnn-recurrent false --lsnn-reset hard --lsnn-adaptive-fraction 0) ;;
                lsnn_ladder_soft)  LSNN_FLAGS=(--lsnn-recurrent false --lsnn-reset soft --lsnn-adaptive-fraction 0) ;;
                lsnn_ladder_rec)   LSNN_FLAGS=(--lsnn-recurrent true --lsnn-reset soft --lsnn-adaptive-fraction 0) ;;
                lsnn_ladder_width) HIDDEN_DIMS=(256 256); LSNN_FLAGS=(--lsnn-recurrent true --lsnn-reset soft --lsnn-adaptive-fraction 0) ;;
                lsnn_ladder_adapt) HIDDEN_DIMS=(256 256); LSNN_FLAGS=(--lsnn-recurrent true --lsnn-reset soft --lsnn-adaptive-fraction 0.9) ;;
            esac
            PER_SESSION_ROOT="${CHECKPOINT_BASE}/per_session_${1}_binarized"
            if [ "$EXPERIMENT" = "hkm" ]; then
                echo "ERROR: $1 is bmi only" >&2; exit 1
            fi
            EPOCHS=50; PATIENCE=10; SCHEDULER_PATIENCE=10; CHECKPOINT_INTERVAL=10 ;;
        *) echo "ERROR: select_model pooled_pretrain|pooled|lsnn_pretrain|lsnn|lsnn_ladder_[ff|soft|rec|width|adapt]|per_session|ladder_[filters|filters_soft|rec|width|soft|chip]|speck_delay[_pretrain|2|4|_long|_xlong|_random|_shift]|speck_plan_[prune5|tuned]|speck_delay_learn[_shared]|speck_tap1[_wide|_funnel], got '$1'" >&2; exit 1 ;;
    esac
    EXP_NAME="bmi_iaf_${RESET_TYPE}"   # file prefix train_snn.py uses (bmi_{neuron}_{reset})
    CKPT_PREFIX=""
    FINETUNE_ROOT="${CHECKPOINT_BASE}/${POOL_KIND}_finetuned_binarized_soft"
    if [ "${1#speck_delay}" != "$1" ]; then
        CKPT_PREFIX="speckdelay_"
        FINETUNE_ROOT="${CHECKPOINT_BASE}/speckdelay_${POOL_KIND}_finetuned_binarized"
    fi
    if [ "${1#lsnn}" != "$1" ]; then
        [ "${1#lsnn_ladder}" = "$1" ] && LSNN_FLAGS=()
        EXP_NAME="bmi_lsnn"; CKPT_PREFIX="lsnn_"
        FINETUNE_ROOT="${CHECKPOINT_BASE}/lsnn_${POOL_KIND}_finetuned_binarized"
        TRAIN_ARGS=(
            --config "$CONFIG_YAML"
            --arch lsnn
            --hidden-dims "${HIDDEN_DIMS[@]}"
            --lsnn-out-stages 2
            --lsnn-readout-all true
            --weight-init kaiming
            --binarize-input true
            --batch-size "$BATCH_SIZE"
            --num-workers 0
            --epochs "$EPOCHS"
            --patience "$PATIENCE"
            --scheduler-patience "$SCHEDULER_PATIENCE"
            --checkpoint-interval "$CHECKPOINT_INTERVAL"
            "${LSNN_FLAGS[@]}"
        )
        MODEL_DESC="$1: LSNN hidden=${HIDDEN_DIMS[*]} ${LSNN_FLAGS[*]:-(recurrent, 90% adaptive)}, readout from every layer, 2 readout stages, batch=${BATCH_SIZE} epochs<=${EPOCHS}"
        case "$1" in *_pretrain) ;; *) apply_val_frac ;; esac
        return
    fi
    TAU_SYN_ARGS=()
    [ ${#TAU_SYN[@]} -gt 0 ] && TAU_SYN_ARGS=(--tau-syn "${TAU_SYN[@]}")
    TRAIN_ARGS=(
        --config "$CONFIG_YAML"
        --spike-fn multi
        --min-vmem -1
        --use-iaf-squeeze
        --weight-init kaiming
        --neuron-type iaf
        --reset-type "$RESET_TYPE"
        --binarize-input true
        --batch-size "$BATCH_SIZE"
        --num-workers 0
        --epochs "$EPOCHS"
        --patience "$PATIENCE"
        --scheduler-patience "$SCHEDULER_PATIENCE"
        --checkpoint-interval "$CHECKPOINT_INTERVAL"
        --temporal-decay-stages 2
        --hidden-dims "${HIDDEN_DIMS[@]}"
        --spike-thresholds "${THRESHOLDS[@]}"
        "${TAU_SYN_ARGS[@]}"
        "${EXTRA_ARGS[@]}"
    )
    MODEL_DESC="$1: hidden=${HIDDEN_DIMS[*]} reset=${RESET_TYPE} thresholds=${THRESHOLDS[*]} tau_syn=${TAU_SYN[*]:-none}${EXTRA_ARGS[*]:+ ${EXTRA_ARGS[*]}} batch=${BATCH_SIZE} epochs<=${EPOCHS}"
    case "$1" in *_pretrain) ;; *) apply_val_frac ;; esac
}

# Pretraining pool and checkpoint directory; per held-out session with LOSO=1.
# Pool directory names contain "pool", which compute_velocity_scalers.py skips.
pool_dir() {
    if [ "$LOSO" = "1" ]; then
        echo "${DATA_ROOT}/snn_datasets/${EXPERIMENT}/${SUBJECT}/loso_pools/$1"
    else
        echo "${DATA_ROOT}/snn_datasets/${EXPERIMENT}/${SUBJECT}/mua_pretrain_pool"
    fi
}
pretrain_dir() {
    if [ "$LOSO" = "1" ]; then
        echo "${CHECKPOINT_BASE}/${CKPT_PREFIX:-}loso_pretrained_binarized/$1"
    else
        echo "${CHECKPOINT_BASE}/${CKPT_PREFIX:-}full_cohort_pretrained_binarized"
    fi
}

# session_dirs -> SESSION_DIRS: every session of the subject, sorted (array
# task i is session i in every script).
session_dirs() {
    mapfile -t SESSION_DIRS < <(find "$SNN_DATASET_ROOT" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort)
    if [ "${#SESSION_DIRS[@]}" -eq 0 ]; then
        echo "ERROR: no session directories under $SNN_DATASET_ROOT" >&2
        exit 1
    fi
}

# array_session -> SESSION_DIR, SESSION_ID for this array task (exits cleanly
# past the last session).
array_session() {
    session_dirs
    if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
        echo "ERROR: submit as an array, one task per session: --array=0-$((${#SESSION_DIRS[@]} - 1))" >&2
        exit 1
    fi
    if [ "$SLURM_ARRAY_TASK_ID" -ge "${#SESSION_DIRS[@]}" ]; then
        echo "Array task $SLURM_ARRAY_TASK_ID has no session (${#SESSION_DIRS[@]} found); exiting."
        exit 0
    fi
    SESSION_DIR="${SESSION_DIRS[$SLURM_ARRAY_TASK_ID]}"
    SESSION_ID="$(basename "$SESSION_DIR")"
}

# latest_resume_args DIR -> RESUME_ARGS resuming from DIR's newest periodic
# checkpoint, else from its best_model_weights.pth, if any.
latest_resume_args() {
    local latest
    RESUME_ARGS=()
    latest=$(ls "$1"/checkpoint_${EXP_NAME}_epoch*.pth 2>/dev/null | sort -V | tail -1 || true)
    [ -z "$latest" ] && [ -f "$1/best_model_weights.pth" ] && latest="$1/best_model_weights.pth"
    if [ -n "$latest" ]; then
        echo "[resume] $latest"
        RESUME_ARGS=(--resume "$latest")
    fi
}

activate_env() {
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate snn_speck
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
    export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
    export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
}
