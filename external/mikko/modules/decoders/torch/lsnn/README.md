# decoders/torch/lsnn

Fit a recurrent spiking network (LSNN) decoder for regression and make predictions. Implemented in
[PyTorch](https://pytorch.org); the same network as the speech and motor LSNN decoders (Bellec et al., NeurIPS 2018),
sized by `units`.

It takes the same input as the Keras RNNs: a 3D array of windows (samples x window steps x channels). Each window is
read one step (one 40 ms frame) at a time, from a fresh state:

* **Spiking layers** (`n_layers`, `units` neurons each): integrate-and-fire neurons without leak, soft reset, at most
`max_spikes` spikes per step, recurrent within each layer. Each neuron has its own synaptic filter, time constants
log-spaced over `tau_syn_ms` and trained; `adaptive_fraction` of them raise their threshold with every spike, recovering
over `tau_adapt_ms`.
* **Output**: one leaky, non-spiking neuron per target (`out_tau_ms`, trained), reading the spikes of every layer. Its
potential at the window's last step is the prediction.

Training minimizes mean squared error with Adam and surrogate gradients, and stops early on the validation loss
(`patience` epochs, keeping the best epoch's weights), as the Keras decoders do.

__Parameters__

* **method** (str): Required. `fit`, `fit_predict`, or `predict`.
* **units** (int): Required. Neurons per spiking layer.
* **n_layers** (int): Spiking layers. Default = `2`.
* **batch_size** (int): Training batch size. Default = `None` (64).
* **epochs** (int): Maximum number of training epochs. Default = `2048`.
* **learning_rate** (float): Adam learning rate. Default = `0.001`.
* **patience** (int): Epochs without a lower validation loss before stopping. Default = `5`.
* **step_ms** (float): Duration of one window step, to convert the time constants. Default = `40.0`.
* **adaptive_fraction**, **tau_adapt_ms**, **adapt_scale**, **tau_syn_ms**, **out_tau_ms**, **max_spikes**: the
neuron model, defaults `0.9`, `500`, `1.8`, `[20, 1300]`, `40`, `3`.
* **mode** (str): `window` (default): each window read from a fresh state, as the Keras RNNs. `stream`: the windows'
frames are put back into one continuous recording and read without resets; each window's prediction is read at its
last frame, so the look-ahead is the same. Trained on random `stream_chunk_steps` (150) stretches, the first
`stream_burn_in_steps` (25) of each not scored, `stream_batch_size` (16) per batch, each epoch covering the training
frames `stream_coverage` (4) times. On 07_19_11 and the five-word set it was no better than `window`.
* **input_dropout**, **layer_dropout** (float): dropout during training on the input features, and on the spikes each
layer passes to the next layer and to the output. Default = `0.0`.
* **white_noise_sd**, **constant_offset_sd** (float): training-only input noise: white noise on every value, and a
constant offset per window and channel. Default = `0.0`.
* **spike_rate_weight**, **spike_rate_target** (float): penalty on mean spikes per neuron per step above the target.
Default = `0.0` (off), `0.1`.
* **device** (str): `cpu` or `cuda`. Default: `cuda` if available.
* **save_model** (bool): Save the fitted model to `model_pt`. Default = `false`.
* **random_seed** (int): Seed for numpy, random and torch. Default = `1337`.

__Input Pathnames__

* **X_train_npy**, **X_valid_npy**, **X_test_npy** (numpy file): 3D arrays of window features.
* **y_train_npy**, **y_valid_npy**, **y_test_npy** (numpy file): 2D arrays of targets.
* **model_pt** (torch file): A saved model, for `predict`.
* **init_model_pt** (torch file): A saved model to start training from (fine-tuning), e.g. one pretrained on a
larger dataset. Optional.

__Output Pathnames__

* **predict_train_npy**, **predict_valid_npy**, **predict_test_npy** (numpy file): predictions.
* **model_pt** (torch file): the fitted model, when `save_model` is `true`.
