"""
On-Policy Distillation 小樣本驗證腳本
=====================================

驗證邏輯：
1. 在 held-out prompts 上，讓訓練前/後的 student 各自 generate
2. 用 teacher 算 reverse KL → 訓練後應下降
3. 用 teacher 算對 student 生成文本的 avg log-prob → 訓練後應上升
4. 印出生成文本，人眼對比品質

用法：
    python eval_distillation.py \
        --base_model Qwen/Qwen3.5-0.8B-Base \
        --trained_model ./distilled_model \
        --teacher_model Qwen/Qwen3.5-2B \
        --num_prompts 5 \
        --max_new_tokens 64 \
        --num_samples 4
"""

import argparse
import json
import logging
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── 工具函數（與訓練腳本一致）──────────────────────────────

def get_per_token_logprobs(model, input_ids, attention_mask):
    """回傳每個 token 位置的 log probability (batch, seq_len-1)"""
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_logprobs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return token_logprobs


def build_gen_mask(prompt_len, total_len, device):
    """只在 generation 部分為 1 的 mask (total_len-1,)"""
    mask = torch.zeros(total_len - 1, device=device)
    mask[prompt_len - 1:] = 1.0
    return mask


# ── 單次評估：generate → 計算指標 ───────────────────────────

@torch.no_grad()
def evaluate_model(
    student, teacher, tokenizer, prompts,
    max_new_tokens, num_samples, device,
):
    """
    回傳 dict:
      - avg_reverse_kl: 平均 reverse KL（越低越好）
      - avg_teacher_logprob: teacher 對 student 生成的平均 log-prob（越高越好）
      - generations: list[dict] 每個 prompt 的生成文本
    """
    student.eval()

    all_rkl = []
    all_teacher_lp = []
    generations = []

    for idx, prompt in enumerate(prompts):
        # 重複 prompt num_samples 次，一次 forward 做完
        enc = tokenizer(
            [prompt] * num_samples,
            return_tensors="pt", padding=True, truncation=True,
        ).to(device)
        prompt_len = enc.attention_mask[0].sum().item()

        # Student generate
        gen_ids = student.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen_mask_attn = (gen_ids != tokenizer.pad_token_id).long()

        # Student & teacher logprobs
        s_lp = get_per_token_logprobs(student, gen_ids, gen_mask_attn)
        t_lp = get_per_token_logprobs(teacher, gen_ids, gen_mask_attn)

        # Generation mask
        masks = torch.stack([
            build_gen_mask(prompt_len, gen_ids.shape[1], device)
            for _ in range(num_samples)
        ])

        # Reverse KL = student_logprob - teacher_logprob（只算 generation 部分）
        rkl = (s_lp - t_lp) * masks
        active = masks.bool()
        if active.sum() > 0:
            all_rkl.append(rkl[active].mean().item())
            all_teacher_lp.append(t_lp[active].mean().item())

        # 存第一個 sample 的生成文本
        gen_text = tokenizer.decode(
            gen_ids[0][prompt_len:], skip_special_tokens=True,
        )
        generations.append({"prompt_idx": idx, "text": gen_text})

    return {
        "avg_reverse_kl": sum(all_rkl) / len(all_rkl) if all_rkl else float("nan"),
        "avg_teacher_logprob": sum(all_teacher_lp) / len(all_teacher_lp) if all_teacher_lp else float("nan"),
        "generations": generations,
    }


# ── 主程式 ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True,
                        help="訓練前的 student（原始 checkpoint）")
    parser.add_argument("--trained_model", type=str, required=True,
                        help="訓練後的 student（distilled checkpoint）")
    parser.add_argument("--teacher_model", type=str, required=True)
    parser.add_argument("--num_prompts", type=int, default=5,
                        help="用幾個 held-out prompt 做驗證")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=4,
                        help="每個 prompt 取樣幾次（取平均）")
    parser.add_argument("--prompt_offset", type=int, default=200,
                        help="從 GSM8K 的第幾筆開始取 held-out prompts（避開訓練用的前 200 筆）")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default=None,
                        help="（可選）將結果存成 JSON")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    logger.info(f"Device: {device}")

    # ── 載入 held-out prompts ────────────────────────────
    ds = load_dataset("openai/gsm8k", "main", split="train")
    prompts = []
    for i in range(args.prompt_offset, args.prompt_offset + args.num_prompts):
        ex = ds[i]
        prompts.append(
            f"Solve the following math problem step by step.\n\n"
            f"Question: {ex['question']}\n\nSolution:"
        )
    logger.info(f"Held-out prompts: {len(prompts)} (offset={args.prompt_offset})")

    # ── 載入 tokenizer & teacher（只需一份）────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Loading teacher: {args.teacher_model}")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, torch_dtype=torch.bfloat16,
    ).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # ── 評估「訓練前」的 student ──────────────────────────
    logger.info(f"Loading BEFORE student: {args.base_model}")
    student_before = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
    ).to(device)

    logger.info("Evaluating BEFORE training...")
    res_before = evaluate_model(
        student_before, teacher, tokenizer, prompts,
        args.max_new_tokens, args.num_samples, device,
    )
    # 釋放記憶體
    del student_before
    torch.cuda.empty_cache() if device == "cuda" else None

    # ── 評估「訓練後」的 student ──────────────────────────
    logger.info(f"Loading AFTER student: {args.trained_model}")
    student_after = AutoModelForCausalLM.from_pretrained(
        args.trained_model, torch_dtype=torch.bfloat16,
    ).to(device)

    logger.info("Evaluating AFTER training...")
    res_after = evaluate_model(
        student_after, teacher, tokenizer, prompts,
        args.max_new_tokens, args.num_samples, device,
    )
    del student_after
    torch.cuda.empty_cache() if device == "cuda" else None

    # ── 輸出比較結果 ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("  On-Policy Distillation 驗證結果")
    print("=" * 70)

    print(f"\n{'指標':<28} {'訓練前':>12} {'訓練後':>12} {'變化':>12}")
    print("-" * 64)

    rkl_b, rkl_a = res_before["avg_reverse_kl"], res_after["avg_reverse_kl"]
    tlp_b, tlp_a = res_before["avg_teacher_logprob"], res_after["avg_teacher_logprob"]

    rkl_delta = rkl_a - rkl_b
    tlp_delta = tlp_a - tlp_b

    print(f"{'Avg Reverse KL (↓ better)':<28} {rkl_b:>12.4f} {rkl_a:>12.4f} {rkl_delta:>+12.4f}")
    print(f"{'Avg Teacher LogProb (↑ better)':<28} {tlp_b:>12.4f} {tlp_a:>12.4f} {tlp_delta:>+12.4f}")

    # 判斷
    print("\n判定：")
    if rkl_delta < -0.005:
        print("  ✅ Reverse KL 下降 → student 更接近 teacher")
    elif rkl_delta > 0.005:
        print("  ⚠️  Reverse KL 上升 → 訓練可能無效或步數不足")
    else:
        print("  ➖ Reverse KL 變化極小 → 步數可能太少，建議增加訓練量")

    if tlp_delta > 0.005:
        print("  ✅ Teacher log-prob 上升 → teacher 更認同 student 的生成")
    elif tlp_delta < -0.005:
        print("  ⚠️  Teacher log-prob 下降 → 訓練方向可能有誤")
    else:
        print("  ➖ Teacher log-prob 變化極小")

    # 印出生成文本對比
    print("\n" + "=" * 70)
    print("  生成文本對比（每個 prompt 各取 1 個 sample）")
    print("=" * 70)
    for i in range(len(prompts)):
        print(f"\n--- Prompt {i} ---")
        print(f"Q: {prompts[i][:120]}...")
        print(f"\n[BEFORE] {res_before['generations'][i]['text'][:300]}")
        print(f"\n[AFTER]  {res_after['generations'][i]['text'][:300]}")

    # 可選：存 JSON
    if args.output:
        out = {
            "before": {k: v for k, v in res_before.items() if k != "generations"},
            "after":  {k: v for k, v in res_after.items() if k != "generations"},
            "delta_reverse_kl": rkl_delta,
            "delta_teacher_logprob": tlp_delta,
            "generations_before": res_before["generations"],
            "generations_after": res_after["generations"],
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()