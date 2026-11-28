"""Run restartable GPU chunks, with validation/sampling only at each chunk boundary."""

import argparse
import subprocess
import sys

import torch

from ..config import ARTIFACTS, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=float, default=90)
    parser.add_argument("--chunk-batches", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--steps", type=int, default=100000)
    args = parser.parse_args()
    if min(args.minutes, args.chunk_batches, args.batch_size, args.steps) <= 0:
        parser.error("Training limits must be positive")
    directory = ARTIFACTS / "scene"
    while True:
        checkpoint_path = directory / "last.pt"
        record = torch.load(checkpoint_path, map_location="cpu", weights_only=True) if checkpoint_path.exists() else None
        step = record["step"] if record else 0
        used = record.get("total_train_seconds", 0) if record else 0
        remaining = args.minutes * 60 - used
        if (directory / "STOP").exists():
            print(f"Paused at saved batch {step}; STOP file is present.", flush=True)
            return
        if remaining <= 0 or step >= args.steps:
            print(f"Training budget complete at batch {step}, {used / 60:.1f} recorded minutes.", flush=True)
            return
        batches = min(args.chunk_batches - step % args.chunk_batches, args.steps - step)
        command = [
            sys.executable, "-X", "utf8", "-m", "scratch.experiments.coco_train", "--width", "64",
            "--batch-size", str(args.batch_size), "--steps", str(args.steps),
            "--hours", str(remaining / 3600), "--session-batches", str(batches),
            "--save-every", str(args.chunk_batches), "--checkpoint-every", "1", "--text-mode", "qwen",
        ]
        if record:
            command.append("--resume")
        print(f"Chunk starts after batch {step}; at most {batches} new batches, {remaining / 60:.1f} minutes remaining.", flush=True)
        del record
        try:
            result = subprocess.run(command, cwd=ROOT)
        except KeyboardInterrupt:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "STOP").touch()
            print("Pause requested. The current child process will save its batch and exit.", flush=True)
            return
        if result.returncode != 0:
            raise SystemExit(f"Training chunk exited with code {result.returncode}; the last complete batch remains saved.")
        updated = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if updated["step"] <= step:
            raise RuntimeError("No progress in the last chunk; refusing to start another")
        del updated


if __name__ == "__main__":
    main()
