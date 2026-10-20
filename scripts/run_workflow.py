#!/usr/bin/env python3
"""Run base evaluation, Training Hub LoRA SFT, and held-out evaluation."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
sys.path.insert(0, str(PROJECT_ROOT))

from data.prepare_data import SYSTEM_PROMPT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-1.5B-Instruct"),
        help="Hugging Face model ID or local model path.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "generated",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Persistent directory for the adapter/checkpoints and metrics.",
    )
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument(
        "--qlora",
        action="store_true",
        help="Use 4-bit QLoRA to reduce GPU memory use if supported by the image.",
    )
    parser.add_argument(
        "--skip-base-eval",
        action="store_true",
        help="Skip loading/evaluating the base model before fine-tuning.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run `python data/prepare_data.py` first.")
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"No examples found in {path}")
    return rows


def normalize_sql(value: str) -> str:
    value = re.sub(r"```(?:sql)?", "", value, flags=re.IGNORECASE)
    value = value.replace("```", "")
    value = re.sub(r"\s+", " ", value).strip().lower()
    return value.rstrip(";").strip()


def move_to_device(batch: Any, device: Any) -> Any:
    if isinstance(batch, dict):
        return {key: value.to(device) for key, value in batch.items()}
    return batch.to(device)


def generate_sql(model: Any, tokenizer: Any, question: str, torch: Any) -> str:
    encoded = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = encoded if torch.is_tensor(encoded) else encoded["input_ids"]
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[0, input_ids.shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def evaluate(model: Any, tokenizer: Any, examples: list[dict[str, str]], torch: Any) -> dict[str, Any]:
    predictions = []
    correct = 0
    for row in examples:
        prediction = generate_sql(model, tokenizer, row["question"], torch)
        expected = row["expected_sql"]
        matched = normalize_sql(prediction) == normalize_sql(expected)
        correct += int(matched)
        predictions.append(
            {
                "question": row["question"],
                "expected_sql": expected,
                "predicted_sql": prediction,
                "exact_match": matched,
            }
        )
    return {
        "correct": correct,
        "total": len(examples),
        "exact_match": correct / len(examples),
        "predictions": predictions,
    }


def load_base_model(model_path: str, torch: Any, AutoModelForCausalLM: Any, AutoTokenizer: Any):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
    ).to("cuda")
    model.eval()
    return model, tokenizer


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    args = parse_args()
    train_path = args.data_dir / "train.jsonl"
    eval_path = args.data_dir / "eval.jsonl"
    train_rows = load_jsonl(train_path)
    eval_rows = load_jsonl(eval_path)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or PROJECT_ROOT / "outputs" / f"run-{stamp}"
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from training_hub import lora_sft
    except ImportError as exc:
        print(
            "Training dependencies are missing. Use the RHOAI Training Workbench image; "
            "if approved, install requirements.txt.\n"
            f"Import error: {exc}",
            file=sys.stderr,
        )
        return 2

    if not torch.cuda.is_available():
        print("A CUDA GPU is required. Recreate the RHOAI Workbench with one GPU.", file=sys.stderr)
        return 2

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Model: {args.model_path}")
    print(f"Training examples: {len(train_rows)}; evaluation examples: {len(eval_rows)}")
    print(f"Output: {output_dir}")

    baseline = None
    if not args.skip_base_eval:
        print("Evaluating base model...")
        base_model, base_tokenizer = load_base_model(
            args.model_path, torch, AutoModelForCausalLM, AutoTokenizer
        )
        baseline = evaluate(base_model, base_tokenizer, eval_rows, torch)
        del base_model, base_tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Base exact match: {baseline['correct']}/{baseline['total']}")

    training_kwargs: dict[str, Any] = {
        "model_path": args.model_path,
        "data_path": str(train_path),
        "ckpt_output_dir": str(adapter_dir),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "micro_batch_size": 1,
        "max_seq_len": args.max_seq_len,
        "dataset_type": "chat_template",
        "field_messages": "messages",
        "logging_steps": 1,
    }
    if args.qlora:
        training_kwargs["load_in_4bit"] = True
    if hasattr(torch.cuda, "is_bf16_supported"):
        training_kwargs["bf16"] = bool(torch.cuda.is_bf16_supported())

    print("Starting Training Hub LoRA SFT...")
    start = time.monotonic()
    result = lora_sft(**training_kwargs)
    training_seconds = time.monotonic() - start
    if not isinstance(result, dict) or "model" not in result or "tokenizer" not in result:
        raise RuntimeError(
            "Training Hub returned an unexpected result; expected model and tokenizer keys."
        )

    print("Evaluating LoRA model...")
    result["model"].eval()
    adapted = evaluate(result["model"], result["tokenizer"], eval_rows, torch)
    print(f"LoRA exact match: {adapted['correct']}/{adapted['total']}")

    metrics = {
        "model_path": args.model_path,
        "gpu": gpu_name,
        "training_hub_version": package_version("training-hub"),
        "training_examples": len(train_rows),
        "evaluation_examples": len(eval_rows),
        "training_parameters": {
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "max_seq_len": args.max_seq_len,
            "qlora": args.qlora,
        },
        "training_seconds": round(training_seconds, 2),
        "base_model": baseline,
        "lora_model": adapted,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Saved metrics: {metrics_path}")
    print(f"Adapter/checkpoints: {adapter_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
