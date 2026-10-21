#!/usr/bin/env python3
"""Run base evaluation, Training Hub LoRA SFT, and held-out evaluation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
sys.path.insert(0, str(PROJECT_ROOT))

from data.prepare_data import SYSTEM_PROMPT  # noqa: E402

PROFILES = {
    "smoke": {"train_repeat": 1, "num_epochs": 2},
    "contention": {"train_repeat": 8, "num_epochs": 5},
}


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
        help="Persistent directory for run metadata, checkpoints, adapter, and metrics.",
    )
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID"))
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILES),
        default=os.environ.get("TRAINING_PROFILE", "smoke"),
        help="smoke is short; contention runs longer to expose scheduling behavior.",
    )
    parser.add_argument("--train-repeat", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=5)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the latest full Trainer checkpoint in this run directory.",
    )
    parser.add_argument(
        "--resume-if-available",
        action="store_true",
        help="Start a new run when no run config exists; otherwise require and resume its latest checkpoint.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Resume from a specific full Trainer checkpoint directory.",
    )
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def current_git_commit() -> str | None:
    value = os.environ.get("GIT_COMMIT")
    if value:
        return value
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def current_git_worktree_dirty() -> bool | None:
    value = os.environ.get("GIT_WORKTREE_DIRTY")
    if value is not None:
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        return None
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def source_snapshot_sha256() -> str:
    digest = hashlib.sha256()
    paths = (
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "data" / "prepare_data.py",
        PROJECT_ROOT / "scripts" / "run_workflow.py",
    )
    for path in paths:
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def write_expanded_training_data(
    rows: list[dict[str, Any]], path: Path, repeat: int
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for _ in range(repeat):
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows) * repeat


def is_full_trainer_checkpoint(path: Path) -> bool:
    if not path.is_dir():
        return False
    files = {item.name for item in path.iterdir() if item.is_file()}
    has_model = any(
        name in files
        for name in (
            "adapter_model.safetensors",
            "adapter_model.bin",
            "model.safetensors",
            "pytorch_model.bin",
        )
    )
    return has_model and "trainer_state.json" in files and "optimizer.pt" in files


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    try:
        from transformers.trainer_utils import get_last_checkpoint
    except ImportError as exc:
        raise RuntimeError("Transformers is required to locate resumable checkpoints.") from exc
    latest = get_last_checkpoint(str(checkpoint_dir))
    return Path(latest) if latest else None


def trainer_checkpoint_hooks(
    resume_checkpoint: Path | None, shutdown_requested: threading.Event
):
    """Bridge Training Hub's lora_sft call to the underlying Transformers Trainer.

    Training Hub 0.9.2 exposes save_steps but does not expose Trainer's
    resume_from_checkpoint argument or a callback parameter. Its LoRA backend
    constructs TRL's SFTTrainer and calls train() internally. This narrowly
    scoped hook adds a callback and forwards the resume path for that call.
    """
    from transformers import TrainerCallback
    from trl import SFTTrainer

    original_init = SFTTrainer.__init__
    original_train = SFTTrainer.train
    if resume_checkpoint:
        train_parameters = inspect.signature(original_train).parameters.values()
        supports_resume = any(
            parameter.name == "resume_from_checkpoint"
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in train_parameters
        )
        if not supports_resume:
            raise RuntimeError(
                "The installed TRL SFTTrainer.train() does not support resume_from_checkpoint."
            )

    class StopAndCheckpointCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if shutdown_requested.is_set():
                control.should_save = True
                control.should_training_stop = True
            return control

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.add_callback(StopAndCheckpointCallback())

    def patched_train(self, *args, **kwargs):
        if resume_checkpoint is not None:
            kwargs["resume_from_checkpoint"] = str(resume_checkpoint)
        return original_train(self, *args, **kwargs)

    class Hooks:
        def __enter__(self):
            SFTTrainer.__init__ = patched_init
            SFTTrainer.train = patched_train
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            SFTTrainer.__init__ = original_init
            SFTTrainer.train = original_train

    return Hooks()


def termination_signal_handler(shutdown_requested: threading.Event):
    class SignalHandler:
        previous_handlers: dict[int, Any]

        def __enter__(self):
            self.previous_handlers = {}

            def request_shutdown(signum, _frame):
                print(f"Received signal {signum}; checkpointing after the current step.", flush=True)
                shutdown_requested.set()

            for signum in (signal.SIGTERM, signal.SIGINT):
                self.previous_handlers[signum] = signal.signal(signum, request_shutdown)
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            for signum, handler in self.previous_handlers.items():
                signal.signal(signum, handler)

    return SignalHandler()


def runtime_metadata(gpu_name: str, torch: Any) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    resource_names = (
        "CPU",
        "MEMORY",
        "GPU",
    )
    resources = {
        category.lower(): {
            name.lower(): os.environ.get(f"{category}_{name}")
            for name in resource_names
        }
        for category in ("REQUEST", "LIMIT")
    }
    return {
        "git_commit": current_git_commit(),
        "git_worktree_dirty": current_git_worktree_dirty(),
        "source_snapshot_sha256": source_snapshot_sha256(),
        "container_image": os.environ.get("WORKLOAD_IMAGE"),
        "scheduler_name": os.environ.get("SCHEDULER_NAME"),
        "pod": {
            "name": os.environ.get("POD_NAME"),
            "namespace": os.environ.get("POD_NAMESPACE"),
            "node": os.environ.get("NODE_NAME"),
        },
        "resources": resources,
        "gpu": {
            "name": gpu_name,
            "memory_bytes": int(properties.total_memory),
            "device_count": int(torch.cuda.device_count()),
        },
        "software_versions": {
            "training_hub": package_version("training-hub"),
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "trl": package_version("trl"),
            "unsloth": package_version("unsloth"),
        },
    }


def main() -> int:
    args = parse_args()
    train_path = args.data_dir / "train.jsonl"
    eval_path = args.data_dir / "eval.jsonl"
    train_rows = load_jsonl(train_path)
    eval_rows = load_jsonl(eval_path)

    run_id = args.run_id or (
        f"run-{datetime.now(timezone.utc).strftime('%Y%m%dt%H%M%Sz')}-"
        f"{time.time_ns() % 1_000_000:06d}"
    )
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", run_id):
        raise ValueError(
            "run-id must be lowercase alphanumeric text with optional hyphens, "
            "up to 63 characters."
        )
    defaults = PROFILES[args.profile]
    train_repeat = args.train_repeat if args.train_repeat is not None else defaults["train_repeat"]
    num_epochs = args.num_epochs if args.num_epochs is not None else defaults["num_epochs"]
    if train_repeat < 1 or num_epochs < 1 or args.save_steps < 1 or args.save_total_limit < 1:
        raise ValueError("train-repeat, num-epochs, save-steps, and save-total-limit must be positive.")

    output_dir = args.output_dir or PROJECT_ROOT / "outputs" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run-config.json"
    state_path = output_dir / "run-state.json"
    baseline_path = output_dir / "base-evaluation.json"
    expanded_train_path = output_dir / "train-expanded.jsonl"
    expanded_train_examples = write_expanded_training_data(
        train_rows, expanded_train_path, train_repeat
    )

    training_config = {
        "run_id": run_id,
        "profile": args.profile,
        "model_path": args.model_path,
        "training_repeats": train_repeat,
        "num_epochs": num_epochs,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_seq_len": args.max_seq_len,
        "sample_packing": False,
        "qlora": args.qlora,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "source_training_data_sha256": sha256_file(train_path),
        "expanded_training_data_sha256": sha256_file(expanded_train_path),
        "evaluation_data_sha256": sha256_file(eval_path),
        "skip_base_eval": args.skip_base_eval,
        "source_snapshot_sha256": source_snapshot_sha256(),
    }

    if args.resume_from_checkpoint:
        args.resume = True
    should_resume = args.resume or (args.resume_if_available and config_path.exists())
    if should_resume:
        if not config_path.is_file():
            raise FileNotFoundError(f"Cannot resume {run_id}: missing {config_path}.")
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        if saved_config.get("training_config") != training_config:
            raise ValueError(
                "Resume configuration differs from the original run. Keep the model, data, "
                "profile, and training parameters unchanged."
            )
        if args.resume_from_checkpoint:
            resume_checkpoint = args.resume_from_checkpoint.resolve()
        else:
            resume_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if resume_checkpoint is None or not is_full_trainer_checkpoint(resume_checkpoint):
            raise FileNotFoundError(
                f"No full resumable Trainer checkpoint was found under {checkpoint_dir}."
            )
    else:
        if config_path.exists():
            raise FileExistsError(
                f"Run {run_id} already exists. Use --resume with this run ID or choose a new one."
            )
        resume_checkpoint = None
        write_json(
            config_path,
            {
                "created_at": utc_now(),
                "run_id": run_id,
                "git_commit": current_git_commit(),
                "git_worktree_dirty": current_git_worktree_dirty(),
                "source_snapshot_sha256": source_snapshot_sha256(),
                "training_config": training_config,
            },
        )

    old_state = {}
    if state_path.is_file():
        old_state = json.loads(state_path.read_text(encoding="utf-8"))
    attempt = int(old_state.get("attempt", 0)) + 1
    run_started_at = utc_now()
    write_json(
        state_path,
        {
            "run_id": run_id,
            "status": "running",
            "attempt": attempt,
            "attempt_started_at": run_started_at,
            "resume_from_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
            "last_checkpoint": old_state.get("last_checkpoint"),
        },
    )

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
    print(f"Run ID: {run_id}; profile: {args.profile}; attempt: {attempt}")
    print(
        f"Training examples: {expanded_train_examples} effective "
        f"({len(train_rows)} source x {train_repeat}); evaluation examples: {len(eval_rows)}"
    )
    print(f"Output: {output_dir}")

    baseline = None
    if baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        print("Reusing the saved base-model evaluation for this run.")
    elif not args.skip_base_eval:
        print("Evaluating base model...")
        base_model, base_tokenizer = load_base_model(
            args.model_path, torch, AutoModelForCausalLM, AutoTokenizer
        )
        baseline = evaluate(base_model, base_tokenizer, eval_rows, torch)
        write_json(baseline_path, baseline)
        del base_model, base_tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Base exact match: {baseline['correct']}/{baseline['total']}")

    shutdown_requested = threading.Event()
    training_kwargs: dict[str, Any] = {
        "model_path": args.model_path,
        "data_path": str(expanded_train_path),
        "ckpt_output_dir": str(checkpoint_dir),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "num_epochs": num_epochs,
        "learning_rate": args.learning_rate,
        "micro_batch_size": 1,
        "max_seq_len": args.max_seq_len,
        "dataset_type": "chat_template",
        "field_messages": "messages",
        "sample_packing": False,
        "logging_steps": 1,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
    }
    if args.qlora:
        training_kwargs["load_in_4bit"] = True
    if hasattr(torch.cuda, "is_bf16_supported"):
        training_kwargs["bf16"] = bool(torch.cuda.is_bf16_supported())

    print("Starting Training Hub LoRA SFT...")
    start = time.monotonic()
    with termination_signal_handler(shutdown_requested):
        with trainer_checkpoint_hooks(resume_checkpoint, shutdown_requested):
            result = lora_sft(**training_kwargs)
    training_seconds = time.monotonic() - start
    if not isinstance(result, dict) or "model" not in result or "tokenizer" not in result:
        raise RuntimeError(
            "Training Hub returned an unexpected result; expected model and tokenizer keys."
        )

    trainer = result.get("trainer")
    global_step = getattr(getattr(trainer, "state", None), "global_step", None)
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    if shutdown_requested.is_set():
        checkpoint_is_resumable = bool(
            latest_checkpoint and is_full_trainer_checkpoint(latest_checkpoint)
        )
        write_json(
            state_path,
            {
                "run_id": run_id,
                "status": (
                    "preempted"
                    if checkpoint_is_resumable
                    else "preempted_without_checkpoint"
                ),
                "attempt": attempt,
                "attempt_started_at": run_started_at,
                "attempt_finished_at": utc_now(),
                "global_step": global_step,
                "last_checkpoint": str(latest_checkpoint) if checkpoint_is_resumable else None,
            },
        )
        if not checkpoint_is_resumable:
            print(
                "Shutdown was requested, but no full Trainer checkpoint is available to resume.",
                file=sys.stderr,
            )
            return 75
        print(f"Saved resumable checkpoint at {latest_checkpoint}; resubmit with --resume {run_id}.")
        return 75

    print("Evaluating LoRA model...")
    result["model"].eval()
    adapted = evaluate(result["model"], result["tokenizer"], eval_rows, torch)
    print(f"LoRA exact match: {adapted['correct']}/{adapted['total']}")

    metrics = {
        "run_id": run_id,
        "profile": args.profile,
        "attempt": attempt,
        "started_at": run_started_at,
        "finished_at": utc_now(),
        "model_path": args.model_path,
        "runtime": runtime_metadata(gpu_name, torch),
        "training_examples": expanded_train_examples,
        "source_training_examples": len(train_rows),
        "evaluation_examples": len(eval_rows),
        "training_parameters": {
            "num_epochs": num_epochs,
            "training_repeats": train_repeat,
            "learning_rate": args.learning_rate,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "max_seq_len": args.max_seq_len,
            "sample_packing": False,
            "qlora": args.qlora,
            "save_steps": args.save_steps,
            "save_total_limit": args.save_total_limit,
        },
        "training_state": {
            "resumed_from_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
            "global_step": global_step,
            "latest_checkpoint": str(latest_checkpoint) if latest_checkpoint else None,
        },
        "training_seconds": round(training_seconds, 2),
        "base_model": baseline,
        "lora_model": adapted,
    }
    metrics_path = output_dir / "metrics.json"
    write_json(metrics_path, metrics)
    write_json(
        state_path,
        {
            "run_id": run_id,
            "status": "completed",
            "attempt": attempt,
            "attempt_started_at": run_started_at,
            "attempt_finished_at": metrics["finished_at"],
            "global_step": global_step,
            "last_checkpoint": str(latest_checkpoint) if latest_checkpoint else None,
            "metrics": str(metrics_path),
        },
    )
    print(f"Saved metrics: {metrics_path}")
    print(f"Adapter/checkpoints: {checkpoint_dir}")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
