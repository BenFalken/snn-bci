#!/bin/bash
# Consolidated build script - every Docker image the auditory decoding
# pipeline depends on, with the exact tags run_auditory_experiment.py and
# the other orchestration scripts expect. Run this from the mikko-master-3
# repo root.
#
# Usage:
#     bash build_all_images.sh

set -e

echo "======================================================"
echo "librosa_mel_transformer (mikko_test)"
echo "======================================================"
docker build --tag mikko_test -f modules/tasks/auditory/librosa_mel_transformer/Dockerfile .

echo "======================================================"
echo "sklearn_preprocess"
echo "======================================================"
docker build --tag sklearn_preprocess_test -f modules/preprocessing/sklearn_preprocess/Dockerfile .

echo "======================================================"
echo "train_valid_test_split"
echo "======================================================"
docker build --tag train_valid_test_split_test -f modules/preprocessing/train_valid_test_split/Dockerfile .

echo "======================================================"
echo "create_window_features"
echo "======================================================"
docker build --tag create_window_features_test -f modules/preprocessing/create_window_features/Dockerfile .

echo "======================================================"
echo "kalman_filter"
echo "======================================================"
docker build --tag kalman_filter_test -f modules/decoders/kalman_filter/Dockerfile .

echo "======================================================"
echo "wiener_filter"
echo "======================================================"
docker build --tag wiener_filter_test -f modules/decoders/wiener_filter/Dockerfile .

echo "======================================================"
echo "wiener_cascade"
echo "======================================================"
docker build --tag wiener_cascade_test -f modules/decoders/wiener_cascade/Dockerfile .

echo "======================================================"
echo "dense_nn"
echo "======================================================"
docker build --tag dense_nn_test -f modules/decoders/keras/dense_nn/Dockerfile .

echo "======================================================"
echo "gru_rnn"
echo "======================================================"
docker build --tag gru_rnn_test -f modules/decoders/keras/gru_rnn/Dockerfile .

echo "======================================================"
echo "lstm_rnn"
echo "======================================================"
docker build --tag lstm_rnn_test -f modules/decoders/keras/lstm_rnn/Dockerfile .

echo "======================================================"
echo "simple_rnn"
echo "======================================================"
docker build --tag simple_rnn_test -f modules/decoders/keras/simple_rnn/Dockerfile .

echo "======================================================"
echo "lsnn"
echo "======================================================"
docker build --tag lsnn_test -f modules/decoders/torch/lsnn/Dockerfile .

echo "======================================================"
echo "regression_metrics"
echo "======================================================"
docker build --tag regression_metrics_test -f modules/analysis/regression_metrics/Dockerfile .

echo ""
echo "All images built successfully."
docker images | grep -E "mikko_test|sklearn_preprocess_test|train_valid_test_split_test|create_window_features_test|kalman_filter_test|wiener_filter_test|wiener_cascade_test|dense_nn_test|gru_rnn_test|lstm_rnn_test|simple_rnn_test|lsnn_test|regression_metrics_test"