# On-Policy Distillation Implement on PyTorch

This repository is a PyTorch implementation of the method from Thinking Machines Lab:

- Blog post: https://thinkingmachines.ai/blog/on-policy-distillation/
- Loss reference: https://tinker-docs.thinkingmachines.ai/losses#policy-gradient-importance_sampling

The core idea is to train a smaller student model using on-policy trajectories sampled by the student itself, while using a larger teacher model to provide token-level supervision.

## What is implemented

The training script [on_policy_distill.py](on_policy_distill.py) follows this loop:

1. Sample prompts from GSM8K.
2. Student generates rollouts (on-policy).
3. Compute student sampling log-probabilities.
4. Compute teacher log-probabilities on the same trajectories.
5. Build token-level advantages from reverse KL:

    `reverse_kl_t = log pi_s(a_t|s_t) - log pi_T(a_t|s_t)`

    `A_t = -reverse_kl_t`

6. Recompute current student log-probabilities with gradients.
7. Optimize an importance-sampling policy-gradient style objective.

Prompt tokens are masked out; loss is only applied to generated tokens.

## Repository layout

- [on_policy_distill.py](on_policy_distill.py): Main training script for on-policy distillation.
- [eval_distill.py](eval_distill.py): Before/after distillation comparison using reverse KL and teacher log-prob metrics.
- [eval_gsm8k.py](eval_gsm8k.py): GSM8K answer accuracy evaluation.
- [eval_seed_repro.py](eval_seed_repro.py): Log-based reproducibility comparison across runs.
- [run.sh](run.sh): Example commands.
- [pyproject.toml](pyproject.toml): Project metadata and dependencies.
- [distilled_model](distilled_model): Output/checkpoint directory (example artifacts).

## Requirements

- Python 3.12+
- PyTorch with CUDA support recommended
- Hugging Face Transformers + Datasets

Dependencies are declared in [pyproject.toml](pyproject.toml).

## Setup

Using uv (recommended):

```bash
uv sync
```

Or with pip in your own virtual environment:

```bash
pip install torch transformers datasets ipywidgets ninja packaging psutil
```

## Train

Basic example:

```bash
python on_policy_distill.py \
  --student_model Qwen/Qwen3.5-0.8B-Base \
  --teacher_model Qwen/Qwen3.5-2B \
  --dataset gsm8k \
  --num_steps 100 \
  --batch_size 2 \
  --samples_per_prompt 2 \
  --max_new_tokens 64 \
  --lr 1e-6 \
  --output_dir ./distilled_model
```

Useful memory/safety flags (already supported):

```bash
python on_policy_distill.py \
  --gradient_checkpointing \
  --max_seq_len 1024 \
  --save_every 100
```

Notes:

- `--device auto` uses CUDA if available.
- Teacher parameters are frozen.
- If CUDA OOM occurs, training automatically retries with reduced rollout batch size.
- If OOM still occurs at rollout batch size 1, an emergency checkpoint is saved and the script exits.

## Evaluate distillation quality

Compare base student vs distilled student on held-out prompts:

```bash
python eval_distill.py \
  --base_model Qwen/Qwen3.5-0.8B-Base \
  --trained_model ./distilled_model \
  --teacher_model Qwen/Qwen3.5-2B \
  --num_prompts 5 \
  --max_new_tokens 64 \
  --num_samples 4
```

Primary indicators:

- Avg Reverse KL: lower is better.
- Avg Teacher LogProb: higher is better.

## Evaluate downstream GSM8K accuracy

Single model:

```bash
python eval_gsm8k.py \
  --model ./distilled_model \
  --num_problems 50
```

Compare two models:

```bash
python eval_gsm8k.py \
  --model Qwen/Qwen3.5-0.8B-Base \
  --model2 ./distilled_model \
  --num_problems 50
```

## Reproducibility check from logs

```bash
python eval_seed_repro.py train.log train2.log
```

This prints RMSE, mean error, and max error for Loss / Avg Reverse KL / Avg Advantage on overlapping steps.

## Outputs

- Final distilled model is saved to `--output_dir` (default: [distilled_model](distilled_model)).
- Optional periodic checkpoints can be saved with `--save_every`.

## Current limitations

- Prompt source is currently GSM8K-only in [on_policy_distill.py](on_policy_distill.py).
- Distributed training and mixed-precision strategy tuning are not yet integrated.
- Evaluation is lightweight and intended for quick iteration.
- Popular optimization methods are not apply now.

## Citation

If you use this code, please cite the original method source from Thinking Machines Lab:

- https://thinkingmachines.ai/blog/on-policy-distillation/
