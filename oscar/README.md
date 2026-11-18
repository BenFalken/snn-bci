# Oscar code, as it ran

Exact copies (code and configs only) of the three code folders in the Oscar home directory, taken 2026-11-18:

| Folder | What ran from it | Compared with the local copy |
|---|---|---|
| `speechBCI-main` | every speech training round (SNNDecoder: LSNN, Speck R3, S3, V2, V2+C), the paper RNN | SNNDecoder identical in all 169 shared files; 15 Oscar-only configs (incl. `speck_r20_V2C.json`, the channel rankings) are also copied into `speech/SNNDecoder/scripts/` |
| `spike_bmi_lsnn` | the motor LSNN training (leave-one-session-out pre-training, per-session fine-tuning) | 18 files differ, all but `velocity_scalers.json` newer locally: the local `motor/` copy was edited after these runs |
| `spike_bmi_speck_delay` | the motor Speck SNN training (per session, delay taps), Jenkins datasets | 2 files differ; `nwb_conversion/make_hkm_array_subset.py` is NEWER here (it built the Jenkins M1-array dataset) |

The third-party language-model toolkits (SRILM, WeNet) are left out. To reproduce a trained model exactly, use the code
in this folder; the copies elsewhere in the repository are the later, locally edited versions used for evaluation.
