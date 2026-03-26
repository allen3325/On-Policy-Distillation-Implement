"""
On-Policy Distillation — PyTorch 實作
=====================================

根據 Thinking Machines Lab 的文章實作 on-policy distillation：
- Student 模型自行取樣 trajectory（on-policy）
- Teacher 模型對每個 token 給出 log probability 作為 dense reward
- 用 reverse KL 作為 advantage，搭配 importance sampling loss 更新 student

參考：
- https://thinkingmachines.ai/blog/on-policy-distillation/
- https://tinker-docs.thinkingmachines.ai/losses#policy-gradient-importance_sampling

用法：
    python on_policy_distillation.py \
        --student_model Qwen/Qwen2.5-0.5B \
        --teacher_model Qwen/Qwen2.5-1.5B-Instruct \
        --dataset gsm8k \
        --num_steps 100 \
        --batch_size 4 \
        --samples_per_prompt 2 \
        --max_new_tokens 256 \
        --lr 1e-6
"""

import argparse
import logging
import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# 1. 工具函數
# ============================================================================

def get_per_token_logprobs(model, input_ids, attention_mask, target_ids):
    """
    給定一個 model 和一段 token 序列，回傳每個 target token 位置的 log probability。

    記憶體優化：不具現化完整的 log_softmax tensor，
    改用 gather 先取出 target logit 再算 log_softmax（省 vocab_size 維度的記憶體）。

    Args:
        model: HuggingFace CausalLM
        input_ids: (batch, seq_len) — 完整的輸入序列（prompt + generation）
        attention_mask: (batch, seq_len)
        target_ids: (batch, seq_len) — 要算 logprob 的 target token
                    通常 = input_ids shifted left by 1

    Returns:
        logprobs: (batch, seq_len-1) — 每個 position 對應 target token 的 log prob
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    # logits shape: (batch, seq_len, vocab_size)
    logits = outputs.logits[:, :-1, :]  # (batch, seq_len-1, vocab_size)
    targets = target_ids[:, 1:]          # (batch, seq_len-1)

    # 記憶體優化：用 log_softmax + gather 的等價公式
    #   log_softmax(logits)[target] = logits[target] - logsumexp(logits)
    # 這樣不需要具現化完整的 (batch, seq_len, vocab_size) log_probs tensor
    target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    log_normalizer = logits.logsumexp(dim=-1)  # (batch, seq_len-1)
    token_logprobs = target_logits - log_normalizer

    return token_logprobs


def build_generation_mask(prompt_len, total_len, device):
    """
    建立一個 mask，只在 student generation 的部分（非 prompt）為 1。
    我們只對 student 自己生成的 token 計算 loss，prompt 部分不算。

    Args:
        prompt_len: prompt 的 token 數量
        total_len: 整個序列（prompt + generation）的 token 數量

    Returns:
        mask: (total_len - 1,) — 因為 logprobs 比 input_ids 短 1
    """
    mask = torch.zeros(total_len - 1, device=device)
    # logprobs[i] 對應的是 position i 預測 position i+1
    # generation 從 prompt_len 開始，所以 mask 從 prompt_len-1 開始
    # （因為 logprobs 的 index 已經 shift 了 1）
    mask[prompt_len - 1:] = 1.0
    return mask


# ============================================================================
# 2. 資料準備
# ============================================================================

def load_prompts(dataset_name, tokenizer, num_prompts=4000):
    """
    載入 prompt 資料集。這裡用 GSM8K 作為範例。
    你可以替換成任何 prompt 來源。

    Returns:
        list of str: prompt 字串列表
    """
    if dataset_name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="train")
        prompts = []
        for example in ds.select(range(min(num_prompts, len(ds)))):
            # 簡單格式：直接用問題作為 prompt
            prompt = (
                f"Solve the following math problem step by step.\n\n"
                f"Question: {example['question']}\n\n"
                f"Solution:"
            )
            prompts.append(prompt)
        return prompts
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# ============================================================================
# 3. 核心：On-Policy Distillation Training Loop
# ============================================================================

@dataclass
class TrainingMetrics:
    """追蹤訓練過程中的指標"""
    step: int = 0
    avg_reverse_kl: float = 0.0
    avg_advantage: float = 0.0
    avg_loss: float = 0.0
    avg_generation_len: float = 0.0


def on_policy_distillation(
    student_model,
    teacher_model,
    tokenizer,
    prompts: list[str],
    num_steps: int = 100,
    batch_size: int = 4,
    samples_per_prompt: int = 2,
    max_new_tokens: int = 256,
    lr: float = 1e-6,
    temperature: float = 1.0,
    max_grad_norm: float = 1.0,
    device: str = "cuda",
    log_every: int = 5,
    save_every: int = 0,
    output_dir: str = "./distilled_model",
    gradient_checkpointing: bool = False,
    max_seq_len: int = 0,
    enable_thinking: bool = True,
):
    """
    On-Policy Distillation 主迴圈。

    流程（對照文章的 pseudocode）:
        for each step:
            1. 從 prompts 中取一個 batch
            2. Student generate（on-policy 取樣）
            3. 記錄 sampling 時 student 的 log probs
            4. Teacher forward pass → teacher log probs
            5. 計算 advantage = -(student_logprobs - teacher_logprobs)
            6. Student forward pass（有梯度）→ current log probs
            7. importance sampling loss → backprop

    記憶體保護：
        - gradient_checkpointing: 用時間換空間，降低 activation 記憶體
        - max_seq_len: 截斷過長的 generation，避免偶發 OOM
        - OOM 自動降 batch: OOM 時砍半 batch 重試，仍失敗則存模型退出
        - save_every: 定期存 checkpoint，避免 crash 時白訓
    """

    # ── 記憶體優化設定 ──
    if gradient_checkpointing:
        student_model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled for student model")

    optimizer = AdamW(student_model.parameters(), lr=lr)
    prompt_idx = 0
    supports_enable_thinking_arg = True
    warned_enable_thinking_fallback = False

    def _save_and_exit(step):
        """OOM 無法恢復時，存模型並退出"""
        emergency_dir = os.path.join(output_dir, f"emergency-step-{step}")
        os.makedirs(emergency_dir, exist_ok=True)
        student_model.save_pretrained(emergency_dir)
        tokenizer.save_pretrained(emergency_dir)
        logger.error(
            f"OOM at step {step} even with batch_size=1. "
            f"Model saved to {emergency_dir}. "
            f"Consider reducing --max_new_tokens or --max_seq_len."
        )
        raise SystemExit(1)

    def _run_step(expanded_prompts, step):
        """
        執行單一 training step。
        回傳 (success: bool, loss_val, avg_rkl, avg_adv, avg_gen_len)
        OOM 時回傳 success=False。
        """
        nonlocal supports_enable_thinking_arg, warned_enable_thinking_fallback
        try:
            # Prepare chat-formatted model inputs, then tokenize as a batch.
            chat_texts = []
            for prompt in expanded_prompts:
                messages = [{"role": "user", "content": prompt}]
                if supports_enable_thinking_arg:
                    try:
                        text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=enable_thinking,
                        )
                    except TypeError:
                        supports_enable_thinking_arg = False
                        if not warned_enable_thinking_fallback:
                            logger.warning(
                                "tokenizer.apply_chat_template does not support "
                                "enable_thinking; falling back without this argument."
                            )
                            warned_enable_thinking_fallback = True
                        text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                else:
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                chat_texts.append(text)

            prompt_encodings = tokenizer(
                chat_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            prompt_lengths = prompt_encodings.attention_mask.sum(dim=1)

            # ── Step 2: Student generate（on-policy 取樣）──
            student_model.eval()
            with torch.no_grad():
                generated_outputs = student_model.generate(
                    **prompt_encodings,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                )

                if max_seq_len > 0 and generated_outputs.shape[1] > max_seq_len:
                    generated_outputs = generated_outputs[:, :max_seq_len]

                gen_attention_mask = (generated_outputs != tokenizer.pad_token_id).long()

            torch.cuda.empty_cache() if device == "cuda" else None

            # ── Step 3: Sampling 時 student log probs ──
            with torch.no_grad():
                sampling_logprobs = get_per_token_logprobs(
                    student_model, generated_outputs,
                    gen_attention_mask, generated_outputs,
                )

            # ── Step 4: Teacher log probs ──
            with torch.no_grad():
                teacher_logprobs = get_per_token_logprobs(
                    teacher_model, generated_outputs,
                    gen_attention_mask, generated_outputs,
                )

            # ── Step 5: Advantage ──
            with torch.no_grad():
                reverse_kl = sampling_logprobs - teacher_logprobs
                advantages = -reverse_kl

                gen_masks = torch.stack([
                    build_generation_mask(
                        prompt_len=prompt_lengths[i].item(),
                        total_len=generated_outputs.shape[1],
                        device=device,
                    )
                    for i in range(generated_outputs.shape[0])
                ])
                advantages = advantages * gen_masks

            # ── Step 6 & 7: Student forward（有梯度）+ loss ──
            student_model.train()
            optimizer.zero_grad()

            current_logprobs = get_per_token_logprobs(
                student_model, generated_outputs,
                gen_attention_mask, generated_outputs,
            )

            log_ratio = current_logprobs - sampling_logprobs.detach()
            ratio = torch.exp(log_ratio)
            token_losses = ratio * advantages.detach()
            loss = -token_losses.sum()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_grad_norm)
            optimizer.step()

            # ── Metrics ──
            with torch.no_grad():
                active_mask = gen_masks.bool()
                if active_mask.sum().item() > 0:
                    avg_rkl = reverse_kl[active_mask].mean().item()
                    avg_adv = advantages[active_mask].mean().item()
                    avg_gen_len = (
                        generated_outputs.shape[1] - prompt_lengths.float().mean()
                    ).item()
                else:
                    avg_rkl = avg_adv = avg_gen_len = 0.0

            return True, loss.item(), avg_rkl, avg_adv, avg_gen_len

        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            return False, 0.0, 0.0, 0.0, 0.0

    # ================================================================
    # 主迴圈
    # ================================================================
    for step in range(1, num_steps + 1):

        # ── Step 1: 取一個 batch 的 prompts ──
        batch_prompts = []
        for _ in range(batch_size):
            batch_prompts.append(prompts[prompt_idx % len(prompts)])
            prompt_idx += 1

        expanded_prompts = batch_prompts * samples_per_prompt
        current_batch = len(expanded_prompts)

        # ── 嘗試執行，OOM 時砍半 batch 重試 ──
        success = False
        while not success:
            selected = expanded_prompts[:current_batch]
            success, loss_val, avg_rkl, avg_adv, avg_gen_len = _run_step(
                selected, step,
            )

            if success:
                if current_batch < len(expanded_prompts):
                    logger.info(
                        f"Step {step}: recovered with reduced batch "
                        f"({current_batch}/{len(expanded_prompts)} rollouts)"
                    )
                break

            # OOM → 砍半
            new_batch = max(1, current_batch // 2)
            if new_batch == current_batch:
                # 已經是 batch=1 還 OOM → 存模型退出
                _save_and_exit(step)

            logger.warning(
                f"Step {step}: OOM with {current_batch} rollouts, "
                f"retrying with {new_batch}"
            )
            current_batch = new_batch

        # ── Logging ──
        if step % log_every == 0 or step == 1:
            logger.info(
                f"Step {step:4d}/{num_steps} | "
                f"Loss: {loss_val:8.4f} | "
                f"Avg Reverse KL: {avg_rkl:7.4f} | "
                f"Avg Advantage: {avg_adv:7.4f} | "
                f"Avg Gen Len: {avg_gen_len:5.1f}"
            )

        # ── 定期存 checkpoint ──
        if save_every > 0 and step % save_every == 0:
            ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            student_model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            logger.info(f"Checkpoint saved to {ckpt_dir}")

    logger.info("Training complete!")
    return student_model


# ============================================================================
# 4. 主程式
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="On-Policy Distillation")
    # parser.add_argument("--seed_for_verify_loss_stable", type=int, default=None,
    #                     help="固定 seed 來驗證 loss 跟 reverseKL 是否穩定")
    parser.add_argument("--student_model", type=str, default="Qwen/Qwen3.5-0.8B-Base",
                        help="Student model name or path")
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen3.5-2B",
                        help="Teacher model name or path")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        help="Dataset name for prompts")
    parser.add_argument("--num_steps", type=int, default=100,
                        help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Number of unique prompts per step")
    parser.add_argument("--samples_per_prompt", type=int, default=2,
                        help="Number of rollouts per prompt")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Max tokens to generate per rollout")
    parser.add_argument("--lr", type=float, default=1e-6,
                        help="Learning rate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature for student generation")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")
    parser.add_argument("--log_every", type=int, default=5,
                        help="Log metrics every N steps")
    parser.add_argument("--output_dir", type=str, default="./distilled_model",
                        help="Directory to save the trained student model")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cuda', 'cpu', or 'auto'")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (set to int for reproducibility)")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="啟用 gradient checkpointing（用時間換記憶體）")
    parser.add_argument("--max_seq_len", type=int, default=0,
                        help="截斷 generation 的最大序列長度（0=不截斷）。"
                             "建議設為 prompt 平均長度 + max_new_tokens")
    parser.add_argument("--save_every", type=int, default=0,
                        help="每 N 步存一次 checkpoint（0=不存中間 checkpoint）")
    parser.add_argument("--enable_thinking", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="是否傳入 enable_thinking 給 apply_chat_template"
                             "（可用 --enable_thinking 或 --no-enable_thinking）")
    args = parser.parse_args()

    # 決定 device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Using device: {device}")

    # if args.seed is not None:
    #     random.seed(args.seed)
    #     np.random.seed(args.seed)
    #     torch.manual_seed(args.seed)
    #     torch.cuda.manual_seed_all(args.seed)

    #     logger.info(f"Set random seed: {args.seed} (deterministic mode)")

    # ----------------------------------------------------------------
    # 載入 tokenizer
    # ----------------------------------------------------------------
    logger.info(f"Loading tokenizer from {args.student_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ----------------------------------------------------------------
    # 載入 student model（需要梯度）
    # ----------------------------------------------------------------
    logger.info(f"Loading student model: {args.student_model}")
    student_model = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if device == "cuda" else "eager",
    ).to(device)

    # ----------------------------------------------------------------
    # 載入 teacher model（inference only，不需要梯度）
    # ----------------------------------------------------------------
    logger.info(f"Loading teacher model: {args.teacher_model}")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if device == "cuda" else "eager",
    ).to(device)
    teacher_model.eval()
    # 凍結 teacher 的所有參數，節省記憶體
    for param in teacher_model.parameters():
        param.requires_grad = False

    # ----------------------------------------------------------------
    # 載入 prompts
    # ----------------------------------------------------------------
    logger.info(f"Loading prompts from {args.dataset}")
    prompts = load_prompts(args.dataset, tokenizer)
    logger.info(f"Loaded {len(prompts)} prompts")

    # ----------------------------------------------------------------
    # 開始訓練
    # ----------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Starting On-Policy Distillation")
    logger.info(f"  Student: {args.student_model}")
    logger.info(f"  Teacher: {args.teacher_model}")
    logger.info(f"  Steps: {args.num_steps}")
    logger.info(f"  Batch size: {args.batch_size} prompts × {args.samples_per_prompt} samples")
    logger.info(f"  Effective batch: {args.batch_size * args.samples_per_prompt} rollouts/step")
    logger.info(f"  Max new tokens: {args.max_new_tokens}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info("=" * 60)

    trained_student = on_policy_distillation(
        student_model=student_model,
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        prompts=prompts,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        samples_per_prompt=args.samples_per_prompt,
        max_new_tokens=args.max_new_tokens,
        lr=args.lr,
        temperature=args.temperature,
        max_grad_norm=args.max_grad_norm,
        device=device,
        log_every=args.log_every,
        save_every=args.save_every,
        output_dir=args.output_dir,
        gradient_checkpointing=args.gradient_checkpointing,
        max_seq_len=args.max_seq_len,
        enable_thinking=args.enable_thinking,
    )

    # ----------------------------------------------------------------
    # 儲存模型
    # ----------------------------------------------------------------
    logger.info(f"Saving trained student to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    trained_student.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Done!")


if __name__ == "__main__":
    main()