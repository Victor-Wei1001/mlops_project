# tools/m27_drift_check.py
"""
M27: Check robustness towards data drift

This script:
1) Loads training tensors from data/processed/train_data.pt and summarizes
   input length distribution.
2) Loads HuggingFace T5 model + (optional) finetuned weights from
   models/final_model.pt.
3) Runs a small suite of "drifted" inputs (short/long/colloquial/mixed-language
   and prompt mismatch) and records outputs, empty-output rate, and simple
   language heuristics.
4) Writes results to tools/output/m27/m27_results.json and prints a readable
   summary.

Run:
  python tools/m27_drift_check.py --sample_n 1000 --max_new_tokens 64
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _quantiles(values: List[int], qs: List[float]) -> Dict[str, int]:
    if not values:
        return {str(q): 0 for q in qs}

    v = sorted(values)
    n = len(v)
    out: Dict[str, int] = {}

    for q in qs:
        idx = int(round((n - 1) * q))
        idx = max(0, min(n - 1, idx))
        out[str(q)] = v[idx]

    return out


def _basic_stats(lengths: List[int]) -> Dict[str, Any]:
    if not lengths:
        return {"count": 0}

    return {
        "count": len(lengths),
        "min": int(min(lengths)),
        "max": int(max(lengths)),
        "mean": float(sum(lengths) / len(lengths)),
        "q": _quantiles(lengths, [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]),
    }


def _has_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _looks_german(text: str) -> bool:
    if re.search(r"[äöüÄÖÜß]", text):
        return True

    common = [" der ", " die ", " das ", " ich ", " und ", " aber ", " nicht "]
    t = " " + text.lower() + " "
    return any(w in t for w in common)


@dataclass
class DriftCase:
    name: str
    prompt_template: str
    text: str


@dataclass
class DriftResult:
    name: str
    prompt_template: str
    input_text: str
    final_prompt: str
    output: str
    is_empty: bool
    has_chinese: bool
    looks_german: bool


def load_training_length_stats(
    train_pt_path: str,
    sample_n: int,
    seed: int,
) -> Dict[str, Any]:
    if not os.path.exists(train_pt_path):
        raise FileNotFoundError(f"Training data not found: {train_pt_path}")

    data = torch.load(train_pt_path, map_location="cpu")
    attn = data.get("attention_mask", None)
    if attn is None:
        raise KeyError("train_data.pt missing 'attention_mask' key")

    n_total = int(attn.shape[0])
    idxs = list(range(n_total))
    random.Random(seed).shuffle(idxs)
    idxs = idxs[: min(sample_n, n_total)]

    lengths: List[int] = []
    for i in idxs:
        lengths.append(int(attn[i].sum().item()))

    return {
        "train_pt_path": train_pt_path,
        "n_total": n_total,
        "n_sampled": int(len(idxs)),
        "length_stats_attention_mask": _basic_stats(lengths),
    }


def load_model_and_tokenizer(
    model_name: str,
    finetuned_path: str,
    device: str,
) -> Tuple[Any, Any, bool, Dict[str, int]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    used_finetuned = False
    missing_unexpected = {"missing": 0, "unexpected": 0}

    if os.path.exists(finetuned_path):
        try:
            state = torch.load(finetuned_path, map_location="cpu")

            # If training saved keys prefixed with "t5.",
            # strip the prefix to match HuggingFace model weights.
            if isinstance(state, dict) and any(
                k.startswith("t5.") for k in state.keys()
            ):
                state = {k.replace("t5.", "", 1): v for k, v in state.items()}

            missing, unexpected = model.load_state_dict(state, strict=False)
            missing_unexpected = {
                "missing": len(missing),
                "unexpected": len(unexpected),
            }
            used_finetuned = True
        except Exception:
            used_finetuned = False

    model.to(device)
    model.eval()
    return model, tokenizer, used_finetuned, missing_unexpected


def run_drift_suite(
    model: Any,
    tokenizer: Any,
    device: str,
    cases: List[DriftCase],
    max_new_tokens: int,
) -> List[DriftResult]:
    results: List[DriftResult] = []

    for c in cases:
        final_prompt = c.prompt_template.format(text=c.text)
        inputs = tokenizer(
            final_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
                num_beams=1,
            )

        out = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        is_empty = out == ""

        results.append(
            DriftResult(
                name=c.name,
                prompt_template=c.prompt_template,
                input_text=c.text,
                final_prompt=final_prompt,
                output=out,
                is_empty=is_empty,
                has_chinese=_has_chinese(out),
                looks_german=_looks_german(out),
            )
        )

    return results


def build_default_cases() -> List[DriftCase]:
    """
    Two axes of drift:
    - Prompt drift (different instruction formats)
    - Input drift (short/long/colloquial/mixed-language)
    """
    base_text = "I love studying in Denmark, but winter is challenging."

    very_short = "Nice!"
    colloquial = "Denmark is kinda cool, but man, winter hits different lol."
    mixed_lang = "I love studying in Denmark, aber der Winter ist schwierig."
    non_english = (
        "Ich studiere gern in Dänemark, "
        "aber der Winter ist schwierig."
    )

    long_text = " ".join([base_text] * 30)

    prompt_variants = [
        ("prompt_translate_colon", "translate: {text}"),
        ("prompt_en2zh_explicit", "translate English to Chinese: {text}"),
        ("prompt_no_instruction", "{text}"),
    ]

    inputs = [
        ("in_distribution_standard", base_text),
        ("drift_short", very_short),
        ("drift_colloquial", colloquial),
        ("drift_long", long_text),
        ("drift_mixed_language", mixed_lang),
        ("drift_non_english", non_english),
    ]

    cases: List[DriftCase] = []
    for p_name, p_tmpl in prompt_variants:
        for i_name, txt in inputs:
            cases.append(
                DriftCase(
                    name=f"{p_name}__{i_name}",
                    prompt_template=p_tmpl,
                    text=txt,
                )
            )

    return cases


def summarize_results(results: List[DriftResult]) -> Dict[str, Any]:
    if not results:
        return {"count": 0}

    total = len(results)
    empty_rate = sum(1 for r in results if r.is_empty) / total
    chinese_rate = sum(1 for r in results if r.has_chinese) / total
    german_rate = sum(1 for r in results if r.looks_german) / total

    by_prompt: Dict[str, Dict[str, Any]] = {}
    for r in results:
        by_prompt.setdefault(
            r.prompt_template,
            {"count": 0, "empty": 0, "chinese": 0, "german": 0},
        )
        by_prompt[r.prompt_template]["count"] += 1
        by_prompt[r.prompt_template]["empty"] += int(r.is_empty)
        by_prompt[r.prompt_template]["chinese"] += int(r.has_chinese)
        by_prompt[r.prompt_template]["german"] += int(r.looks_german)

    for v in by_prompt.values():
        c = max(1, v["count"])
        v["empty_rate"] = v["empty"] / c
        v["chinese_rate"] = v["chinese"] / c
        v["german_rate"] = v["german"] / c

    return {
        "count": total,
        "empty_rate": empty_rate,
        "chinese_rate": chinese_rate,
        "german_rate": german_rate,
        "by_prompt_template": by_prompt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pt", default="data/processed/train_data.pt")
    parser.add_argument(
        "--model_name",
        default=os.getenv("HF_MODEL_NAME", "t5-small"),
    )
    parser.add_argument(
        "--finetuned_path",
        default=os.getenv("FINETUNED_WEIGHTS", "models/final_model.pt"),
    )
    parser.add_argument("--sample_n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--out_dir", default="tools/output/m27")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_stats = load_training_length_stats(
        args.train_pt,
        args.sample_n,
        args.seed,
    )

    model, tokenizer, used_finetuned, mu = load_model_and_tokenizer(
        model_name=args.model_name,
        finetuned_path=args.finetuned_path,
        device=device,
    )

    cases = build_default_cases()
    results = run_drift_suite(
        model=model,
        tokenizer=tokenizer,
        device=device,
        cases=cases,
        max_new_tokens=args.max_new_tokens,
    )
    summary = summarize_results(results)

    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "device": device,
        "model_name": args.model_name,
        "used_finetuned_weights": used_finetuned,
        "finetuned_path": args.finetuned_path,
        "load_state_dict_missing_unexpected": mu,
        "train_length_distribution": train_stats,
        "drift_suite_summary": summary,
        "examples": [asdict(r) for r in results[:10]],
        "n_total_cases": len(results),
    }

    _ensure_dir(args.out_dir)
    out_json = os.path.join(args.out_dir, "m27_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n=== M27 Drift Check Summary ===")
    print(f"Device: {device}")
    print(f"Model: {args.model_name}")
    print(
        "Finetuned weights used: "
        f"{used_finetuned}  "
        f"(missing={mu['missing']} unexpected={mu['unexpected']})"
    )

    ls = train_stats["length_stats_attention_mask"]
    p95 = ls.get("q", {}).get("0.95")
    print(
        f"Train input length stats (sample={train_stats['n_sampled']}): "
        f"mean={ls.get('mean'):.2f}, min={ls.get('min')}, p95={p95}, "
        f"max={ls.get('max')}"
    )
    print(
        f"Drift suite: cases={summary['count']}, "
        f"empty_rate={summary['empty_rate']:.2f}, "
        f"chinese_rate={summary['chinese_rate']:.2f}, "
        f"german_rate={summary['german_rate']:.2f}"
    )
    print(f"Saved JSON: {out_json}")

    print("\n--- Sample outputs (one per prompt template) ---")
    shown = set()
    for r in results:
        if r.prompt_template in shown:
            continue

        shown.add(r.prompt_template)
        out_preview = r.output if r.output else "[EMPTY]"
        input_preview = r.input_text[:120]
        if len(r.input_text) > 120:
            input_preview += "..."

        output_preview = out_preview[:200]
        if len(out_preview) > 200:
            output_preview += "..."

        print(f"\nPrompt template: {r.prompt_template}")
        print(f"Input: {input_preview}")
        print(f"Output: {output_preview}")

    print("\nDone.")


if __name__ == "__main__":
    main()
