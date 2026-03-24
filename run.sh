python on_policy_distill.py \
    --student_model Qwen/Qwen3.5-0.8B-Base \
    --teacher_model Qwen/Qwen3.5-2B \
    --num_steps 2000 \
    --batch_size 2 \
    --max_new_tokens 64


python eval_distillation.py \
    --base_model Qwen/Qwen3.5-0.8B-Base \
    --trained_model ./distilled_model \
    --teacher_model Qwen/Qwen3.5-2B \
    --num_prompts 5 \
    --max_new_tokens 64 \
    --num_samples 4