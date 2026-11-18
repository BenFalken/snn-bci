# Oscar code, as it ran

Exact copies (code and configs only) of the three code folders in the Oscar home directory, taken 2026-11-18:

| Folder | What ran from it | Compared with the local copy |
|---|---|---|
| `speechBCI-main` | every speech training round (SNNDecoder: LSNN, Speck R3, S3, V2, V2+C), the paper RNN | SNNDecoder identical in all 169 shared files; 15 Oscar-only configs (incl. `speck_r20_V2C.json`, the channel rankings) are also copied into `speech/SNNDecoder/scripts/` |
| `spike_bmi_lsnn` | the motor LSNN training (leave-one-session-out pre-training, per-session fine-tuning) | 18 files differ, all but `velocity_scalers.json` newer locally: the local `motor/` copy was edited after these runs |
| `spike_bmi_speck_delay` | the motor Speck SNN training (per session, delay taps), Jenkins datasets | 2 files differ; `nwb_conversion/make_hkm_array_subset.py` is NEWER here (it built the Jenkins M1-array dataset) |

The third-party language-model toolkits (SRILM, WeNet) are left out. The only edit: the cluster login in paths is
written as `$USER` (expanded where it is used), and as `$OSCAR_USER` in the fetch scripts' ssh address. To reproduce a trained model exactly, use the code
in this folder; the copies elsewhere in the repository are the later, locally edited versions used for evaluation.

## The velocity scalers (checked 2026-11-18)

`spike_bmi_lsnn/main/snn_training/velocity_scalers.json` differs from the other copies: it was computed over 72 Indy
and 18 Loco sessions, the others over 73 and 19 (Indy range -284.22 to 314.13 mm/s against -284.38 to 315.68). This
does not affect any result: every checkpoint stores the scaler it was trained with (`velocity_lo`, `velocity_hi`,
`velocity_margin` in its args), and evaluation (`motor/main/inference/decoder_eval.py`, `load_snn_model`) unscales with
that stored value, not with the JSON file. The LSNN checkpoints carry the 72/18-session values, the Speck checkpoints
the 73/19-session values, each matching its own training.
