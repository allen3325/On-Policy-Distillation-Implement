"""
GSM8K 準確率評估腳本
====================

從 GSM8K test set 取題目，讓模型 generate 解答，
提取最終數字答案與標準答案比對，算正確率。

用法：
    # 評估訓練前
    python eval_gsm8k.py --model Qwen/Qwen3.5-0.8B-Base --num_problems 50

    # 評估訓練後
    python eval_gsm8k.py --model ./distilled_model --num_problems 50

    # 對比兩個模型
    python eval_gsm8k.py \
        --model Qwen/Qwen3.5-0.8B-Base \
        --model2 ./distilled_model \
        --num_problems 50
"""

import argparse
import logging
import math
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── 答案提取 ─────────────────────────────────────────────

def extract_gold_answer(answer_text: str) -> str:
    """
    GSM8K 的 answer 欄位格式為 "...#### 72"，
    提取 #### 後面的數字作為標準答案。
    """
    match = re.search(r"####\s*(.+)", answer_text)
    if match:
        return normalize_number(match.group(1).strip())
    return ""


def extract_model_answer(text: str) -> str:
    """
    從模型生成文本中提取最終答案。
    嘗試多種常見格式：
      - "The answer is 42"
      - "#### 42"
      - "= 42" (最後一個等號後)
      - 最後出現的獨立數字
    """
    # 優先抓 #### 格式
    match = re.search(r"####\s*(.+?)(?:\s*$|\n)", text)
    if match:
        return normalize_number(match.group(1).strip())

    # "the answer is ..." 格式
    match = re.search(r"[Tt]he\s+(?:final\s+)?answer\s+is[:\s]*([^\n.]+)", text)
    if match:
        return normalize_number(match.group(1).strip())

    # "**X**" 粗體格式（常見於 instruct 模型）
    matches = re.findall(r"\*\*\s*([\d,.\-]+)\s*\*\*", text)
    if matches:
        return normalize_number(matches[-1])

    # \\boxed{X} 格式
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return normalize_number(match.group(1).strip())

    # 最後一個等號後的數字
    matches = re.findall(r"=\s*([\d,.\-]+)", text)
    if matches:
        return normalize_number(matches[-1])

    # fallback：文本中最後出現的數字
    matches = re.findall(r"([\d,]+\.?\d*)", text)
    if matches:
        return normalize_number(matches[-1])

    return ""


def normalize_number(s: str) -> str:
    """移除逗號、多餘空白，統一數字格式方便比對。"""
    s = s.replace(",", "").replace("$", "").replace("%", "").strip()
    s = s.rstrip(".")
    # 去掉尾部的 .0 / .00
    try:
        val = float(s)
        if not math.isfinite(val):
            return s
        if val == int(val):
            return str(int(val))
        return str(val)
    except (ValueError, OverflowError):
        return s


# ── 評估單一模型 ─────────────────────────────────────────

@torch.no_grad()
def evaluate_gsm8k(model, tokenizer, problems, max_new_tokens, device, batch_size=8):
    """
    回傳 dict:
      - accuracy: 正確率
      - correct: 答對幾題
      - total: 總題數
      - details: list[dict] 每題的詳細結果
    """
    model.eval()
    correct = 0
    details = []

    prompts = [
        f"Solve the following math problem step by step.\n\nQuestion: {ex['question']}\n\nSolution:"
        for ex in problems
    ]
    golds = [extract_gold_answer(ex["answer"]) for ex in problems]

    current_batch_size = batch_size
    start = 0
    processed = 0
    pbar = tqdm(total=len(problems), desc="Evaluating", unit="problem")

    while start < len(problems):
        end = min(start + current_batch_size, len(problems))
        batch_prompts = prompts[start:end]
        batch_golds = golds[start:end]

        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        try:
            gen_ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,           # greedy → 確定性結果
                pad_token_id=tokenizer.pad_token_id,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and current_batch_size > 1:
                current_batch_size = max(1, current_batch_size // 2)
                torch.cuda.empty_cache()
                logger.warning(f"OOM! Reducing batch_size to {current_batch_size} and retrying...")
                del enc
                continue  # retry same `start` with smaller batch
            raise

        prompt_lens = enc.input_ids.shape[1]
        for j, (gold, gen_id) in enumerate(zip(batch_golds, gen_ids)):
            gen_text = tokenizer.decode(gen_id[prompt_lens:], skip_special_tokens=True)
            pred = extract_model_answer(gen_text)
            is_correct = (pred == gold)
            if is_correct:
                correct += 1

            i = start + j
            details.append({
                "idx": i,
                "question": problems[i]["question"][:80],
                "gold": gold,
                "pred": pred,
                "correct": is_correct,
                "generation": gen_text[:200],
            })

        batch_len = end - start
        processed += batch_len
        pbar.update(batch_len)
        start = end
        if processed % 10 == 0 or processed == len(problems):
            logger.info(f"  Progress: {processed}/{len(problems)} | Running acc: {correct/processed:.1%}")

    pbar.close()
    if current_batch_size != batch_size:
        logger.info(f"  Final effective batch_size: {current_batch_size} (started at {batch_size})")

    return {
        "accuracy": correct / len(problems) if problems else 0,
        "correct": correct,
        "total": len(problems),
        "details": details,
    }


# ── 載入模型的 helper ────────────────────────────────────

def load_model(model_path, device):
    logger.info(f"Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
    ).to(device).eval()
    return model


# ── 印出結果 ─────────────────────────────────────────────

def print_results(label, res, show_errors=5):
    print(f"\n{'─' * 50}")
    print(f"  {label}")
    print(f"{'─' * 50}")
    print(f"  正確率: {res['accuracy']:.1%}  ({res['correct']}/{res['total']})")

    # 印出部分錯誤題目
    errors = [d for d in res["details"] if not d["correct"]]
    if errors:
        print(f"\n  錯誤範例（最多 {show_errors} 題）:")
        for d in errors[:show_errors]:
            print(f"    #{d['idx']} | gold={d['gold']} | pred={d['pred']}")
            print(f"         Q: {d['question']}...")
            print(f"         A: {d['generation'][:120]}...")
    else:
        print("  全部答對！")


# ── 主程式 ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GSM8K 準確率評估")
    parser.add_argument("--model", type=str, required=True,
                        help="主要模型（或訓練前模型）")
    parser.add_argument("--model2", type=str, default=None,
                        help="（可選）第二個模型，自動做對比")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="指定 tokenizer（預設同 --model）")
    parser.add_argument("--num_problems", type=int, default=50,
                        help="評估幾題（GSM8K test set 共 1319 題；設為 -1 跑全部）")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="每次 generate 幾題（越大越快，但需要更多 GPU memory）")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    logger.info(f"Device: {device}")

    # 載入資料
    logger.info("Loading GSM8K test set...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    n = len(ds) if args.num_problems <= 0 else min(args.num_problems, len(ds))
    problems = list(ds.select(range(n)))
    logger.info(f"Evaluating on {len(problems)} problems")

    # 載入 tokenizer（decoder-only batched generation 需要 left padding）
    tok_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tok_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── 評估 model 1 ──
    model1 = load_model(args.model, device)
    logger.info(f"Evaluating: {args.model}")
    res1 = evaluate_gsm8k(model1, tokenizer, problems, args.max_new_tokens, device, args.batch_size)
    del model1
    torch.cuda.empty_cache() if device == "cuda" else None
    print_results(args.model, res1)

    # ── 評估 model 2（如有）──
    if args.model2:
        model2 = load_model(args.model2, device)
        logger.info(f"Evaluating: {args.model2}")
        res2 = evaluate_gsm8k(model2, tokenizer, problems, args.max_new_tokens, device, args.batch_size)
        del model2
        torch.cuda.empty_cache() if device == "cuda" else None
        print_results(args.model2, res2)

        # 對比
        delta = res2["accuracy"] - res1["accuracy"]
        print(f"\n{'=' * 50}")
        print(f"  對比結果")
        print(f"{'=' * 50}")
        print(f"  {args.model:<30} {res1['accuracy']:.1%}")
        print(f"  {args.model2:<30} {res2['accuracy']:.1%}")
        print(f"  {'差異':<30} {delta:+.1%}")
        if delta > 0.01:
            print(f"  ✅ 訓練後正確率提升")
        elif delta < -0.01:
            print(f"  ⚠️  訓練後正確率下降")
        else:
            print(f"  ➖ 變化不顯著（步數可能不夠）")


if __name__ == "__main__":
    main()