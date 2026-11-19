# GRU reference decoders (Wang et al. 2024)

The AEGRU of Wang, Wang & Liu (BioCAS 2024, arXiv:2410.22283), first in R² in the BioCAS 2024 neural-decoding
challenge, and its plain-GRU ablation, re-implemented from the paper (no code was released).

| file | does |
|---|---|
| `model.py` | AEGRU (FC 32 + batch norm → GRU 32 → FC 2; training-only variance/reconstruction branch) and plain GRU |
| `data.py` | inputs (WS × N sub-window spike counts, log∘softplus) and splits for both protocols |
| `run_gru.py` | trains one model on one session (50 epochs, Adam 1e-3, weight decay 1e-3, cosine), tests the last epoch |
| `submit_gru.sh`, `run_gru_array.sbatch` | Oscar: step 1 (their protocol, 6 sessions × 5 seeds) and step 2 (ours, 45 sessions × 3 seeds) |
| `score_gru.py` | scores step 2 on the rows `results/final_v2` scored, writes `gru_metrics.json` next to `combined_metrics.json` |

**Protocols.** *neurobench*: the raw `.mat`, all channels (loco 192), binary 4 ms bins (units merged), velocity =
gradient of `cursor_pos`, reach segments split 50/25/25. *ours*: the 96 M1 channels' spike counts, `task_data`
velocity, the shared chronological boundary (last 10 % test) and the LSTM's purge gap.

**Choices the paper leaves open:** batch 256; velocity z-scored on the training split; Poisson weight w_x = 1
(w_v = 1); the GRU gets the sampled latent in training and its mean at inference; WS = 20, N = 5 (their default);
no pruning or quantization (their unpruned AEGRU: R² 0.70). A local check on indy_20170131_02 found batch size
(256–4096), w_x (0.1, 1) and binary vs count inputs within 0.02 of each other on validation.
