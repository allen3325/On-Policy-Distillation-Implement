python on_policy_distill.py \
    --student_model Qwen/Qwen3.5-0.8B-Base \
    --teacher_model Qwen/Qwen3.5-2B \
    --num_steps 2000 \
    --batch_size 2 \
    --max_new_tokens 256 \
    --samples_per_prompt 2

python eval_distillation.py \
    --base_model Qwen/Qwen3.5-0.8B-Base \
    --trained_model ./distilled_model \
    --teacher_model Qwen/Qwen3.5-2B \
    --num_prompts 5 \
    --max_new_tokens 64 \
    --num_samples 4

python eval_gsm8k.py \
        --model Qwen/Qwen3.5-0.8B-Base \
        --model2 ./distilled_model \
        --num_problems -1 \
        --batch_size 64

lm-eval run \
    --model hf \
    --model_args pretrained=Qwen/Qwen3.5-0.8B-Base \
    --tasks gsm8k \
    --num_fewshot 4 \
    --gen_kwargs max_new_tokens=2048 do_sample=False \
    --log_samples \
    --output_path Qwen3.5-0.8B-Base.log \
    --show_config \
    --batch_size auto \
    --seed 42
