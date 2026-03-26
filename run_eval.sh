#!/usr/bin/env bash
set -euo pipefail

echo "[1/3] Running distilled_model ..."
lm-eval run \
    --model hf \
    --model_args pretrained=/home/os-chewei.chang/Projects/qat/naive/distilled_model \
    --tasks gsm8k_custom \
    --log_samples \
    --batch_size 64 \
    --output_path ./outputs/distilled_model/ \
    --include_path /home/os-chewei.chang/Projects/qat/lm-evaluation-harness/gsm8k_custom/ \
    --show_config \
    --seed 42

echo "[2/3] Running Qwen3.5-0.8B-Base ..."
lm-eval run \
    --model hf \
    --model_args pretrained=Qwen/Qwen3.5-0.8B-Base \
    --tasks gsm8k_custom \
    --log_samples \
    --batch_size 64 \
    --output_path ./outputs/Qwen3.5-0.8B-Base/ \
    --include_path /home/os-chewei.chang/Projects/qat/lm-evaluation-harness/gsm8k_custom/ \
    --show_config \
    --seed 42

echo "[3/3] Running Qwen3.5-2B ..."
lm-eval run \
    --model hf \
    --model_args pretrained=Qwen/Qwen3.5-2B \
    --tasks gsm8k_custom \
    --log_samples \
    --batch_size 32 \
    --output_path ./outputs/Qwen3.5-2B/ \
    --include_path /home/os-chewei.chang/Projects/qat/lm-evaluation-harness/gsm8k_custom/ \
    --show_config \
    --seed 42

echo "All runs completed successfully."