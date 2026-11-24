"""
specksim with hard reset imposed, against our chip model (snndec.chip.run_events), on chip/specksim_compare.py's
network and events. specksim's source is compiled inside samna, so the reset is added from outside: each input spike
goes through specksim's own convolution and integrate-and-fire layer (forward_single, forward), and the neurons that
fired are set to zero (get_v_mem / set_v_mem). In these fully connected layers each input spike reaches each neuron
once, so this is exactly a reset to zero at the spike. Spikes are passed on in neuron order, as ours.
    python3 chip/specksim_compare.py && python3 chip/specksim_hard.py
Result (2026-11-24, samna 0.48): identical spike trains, all 3,000 steps, every layer (weights and thresholds are whole
numbers, so float32 is exact). In specksim's own spike order: same rate (48.87 vs 48.85 per step), correlation 0.993.
"""
