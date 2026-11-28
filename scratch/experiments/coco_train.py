"""Time-bounded training of a scene diffusion model entirely from random weights."""

import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
import random
import signal
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from ..config import ARTIFACTS, DATA, tokenize
from .photo_diffusion import PhotoUNet, configure_cuda, noise_schedule, sample
from ..train import save_checkpoint, save_json

PROMPTS = (
    "a group of people cooking food on a grill outdoors",
    "a group of friends sitting at a table eating dinner",
    "people walking on a beach near the ocean",
    "a cat lying on a couch in a room",
)


def capture_random_state(rng):
    return {
        "numpy": rng.bit_generator.state, "python": random.getstate(),
        "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_random_state(state, rng):
    rng.bit_generator.state = state["numpy"]
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def repair_batch_log(path, checkpoint):
    if checkpoint is None:
        return
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            if row["step"] <= checkpoint["step"]:
                rows.append(row)
    if not rows or rows[-1]["step"] < checkpoint["step"]:
        rows.append({"step": checkpoint["step"], "loss": checkpoint.get("last_loss"),
                     "seconds": checkpoint.get("total_train_seconds", 0), "recovered_from_checkpoint": True})
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def encode_caption(caption, vocabulary, length=48):
    ids = [2] + [vocabulary.get(word, 1) for word in tokenize(caption)[:length - 1]]
    return ids + [0] * (length - len(ids))


def build_vocabulary(rows):
    counts = Counter(word for row in rows for caption in row["captions"] for word in tokenize(caption))
    words = sorted(counts, key=lambda word: (-counts[word], word))[:11997]
    return {"<pad>": 0, "<unk>": 1, "<start>": 2, **{word: i + 3 for i, word in enumerate(words)}}


def caption_array(rows, vocabulary):
    return [torch.tensor([encode_caption(text, vocabulary) for text in row["captions"]], dtype=torch.long) for row in rows]


def dataset_hash(directory, size):
    digest = sha256()
    for name in ("train.json", "val.json", "test.json", f"train-{size}.npy", f"val-{size}.npy", f"test-{size}.npy"):
        digest.update(name.encode())
        with (directory / name).open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def training_batch(images, captions, batch_size, rng, device):
    indices = rng.integers(0, len(images), size=batch_size)
    photos = torch.from_numpy(images[indices].copy()).to(device, dtype=torch.float32).div_(127.5).sub_(1)
    if isinstance(captions, np.ndarray):
        choices = rng.integers(captions.shape[1], size=batch_size)
        tokens = torch.from_numpy(captions[indices, choices].copy()).to(device)
    else:
        tokens = torch.stack([captions[index][rng.integers(len(captions[index]))] for index in indices]).to(device)
    return photos, tokens


def diffusion_loss(model, images, tokens, alphas, generator=None, drop_condition=False):
    times = torch.randint(0, len(alphas), (len(images),), device=images.device, generator=generator)
    noise = torch.randn(images.shape, device=images.device, generator=generator)
    a = alphas[times, None, None, None]
    noisy = a.sqrt() * images + (1 - a).sqrt() * noise
    if drop_condition:
        dropped = torch.rand(len(images), device=images.device) < 0.1
        tokens = tokens.clone()
        tokens[dropped] = 0
        if not model.feature_dim:
            tokens[dropped, 0] = 2
    prediction = model(noisy, times, tokens)
    mse = (prediction.float() - noise).square().flatten(1).mean(1)
    snr = a.flatten() / (1 - a.flatten()).clamp_min(1e-8)
    weight = snr.clamp(max=5) / snr
    return (mse * weight).mean()


@torch.inference_mode()
def validation(model, images, captions, alphas, device):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(2026)
    losses = []
    for start in range(0, min(256, len(images)), 16):
        batch = torch.from_numpy(images[start:start + 16].copy()).to(device, dtype=torch.float32).div_(127.5).sub_(1)
        if isinstance(captions, np.ndarray):
            tokens = torch.as_tensor(captions[start:start + 16, 0].copy(), device=device)
        else:
            tokens = torch.stack([row[0] for row in captions[start:start + 16]]).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = diffusion_loss(model, batch, tokens, alphas, generator=generator)
        losses.append(float(loss))
    return sum(losses) / len(losses)


@torch.inference_mode()
def save_preview(model, vocabulary, directory, step, device, size, preview_features=None):
    if preview_features is not None:
        tokens = torch.as_tensor(preview_features, device=device)
    else:
        tokens = torch.tensor([encode_caption(text, vocabulary) for text in PROMPTS], device=device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        pictures = sample(model, tokens, size=size, steps=1000, seed=2026, guidance=2.0, same_noise=True).float().cpu()
    grid = Image.new("RGB", (size * 4, size))
    for index, picture in enumerate(pictures):
        image = Image.fromarray(np.uint8(picture.permute(1, 2, 0).numpy() * 255))
        grid.paste(image, (size * index, 0))
    grid.resize((size * 4 * 3, size * 3), Image.Resampling.NEAREST).save(directory / f"samples-{step:06d}.png")
    grid.resize((size * 4 * 3, size * 3), Image.Resampling.NEAREST).save(directory / "samples-latest.png")
    save_json(directory / "sample_prompts.json", list(PROMPTS))
    save_json(directory / f"samples-{step:06d}.json", {"method": "DDPM", "steps": 1000, "seed": 2026, "same_noise": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=1.5)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, help="Defaults to the saved batch size on resume, otherwise 32")
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=1,
                        help="Update last.pt every N batches; default saves every batch")
    parser.add_argument("--session-batches", type=int,
                        help="Stop after this many additional batches in the current session")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--text-mode", choices=("scratch", "qwen"), default="qwen")
    args = parser.parse_args()
    if min(args.hours, args.steps, args.save_every, args.checkpoint_every) <= 0 or args.width % 8:
        parser.error("Positive limits and width divisible by eight are required")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.session_batches is not None and args.session_batches < 1:
        parser.error("--session-batches must be positive")
    directory = ARTIFACTS / "scene"
    data = DATA / "coco"
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    size = manifest["size"]
    if (directory / "last.pt").exists() and not args.resume:
        parser.error("Use --resume to continue the existing run")
    checkpoint = torch.load(directory / "last.pt", map_location="cpu", weights_only=True) if args.resume else None
    if (directory / "STOP").exists():
        parser.error("A STOP file is present. Remove artifacts/scene/STOP when ready to resume.")
    if args.batch_size is None:
        args.batch_size = checkpoint["arguments"]["batch_size"] if checkpoint else 32
    if checkpoint and checkpoint["arguments"]["seed"] != args.seed:
        parser.error("Resume must use the original seed")
    if checkpoint and checkpoint["arguments"]["batch_size"] != args.batch_size:
        print(f"Resuming with batch size {args.batch_size}; previous batch size was {checkpoint['arguments']['batch_size']}.", flush=True)
    rows = json.loads((data / "train.json").read_text(encoding="utf-8"))
    val_rows = json.loads((data / "val.json").read_text(encoding="utf-8"))
    vocabulary = checkpoint["vocabulary"] if checkpoint else build_vocabulary(rows)
    fingerprint = dataset_hash(data, size)
    if checkpoint and checkpoint["data_hash"] != fingerprint:
        parser.error("Dataset changed since the previous checkpoint")
    if not torch.cuda.is_available():
        parser.error("This training command requires CUDA PyTorch")
    device = torch.device("cuda")
    configure_cuda()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    feature_metadata = json.loads((data / "qwen_features.json").read_text()) if args.text_mode == "qwen" else None
    if feature_metadata:
        for split in ("train", "val", "test"):
            actual = sha256((data / (split + ".json")).read_bytes()).hexdigest()
            if feature_metadata[split]["captions_sha256"] != actual:
                raise ValueError(f"Stale {split} language features")
    feature_dim = feature_metadata["train"]["width"] if feature_metadata else 0
    config = checkpoint["config"] if checkpoint else {
        "vocab_size": 3 if feature_dim else len(vocabulary), "width": args.width,
        "length": 8 if feature_dim else 48, "feature_dim": feature_dim,
    }
    if bool(config.get("feature_dim")) != bool(feature_dim):
        raise ValueError("Resume text mode must match the checkpoint")
    args.width = config["width"]
    model = PhotoUNet(**config).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    ema = deepcopy(model).eval().requires_grad_(False)
    if checkpoint:
        ema.load_state_dict(checkpoint["ema"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    images = np.load(data / f"train-{size}.npy", mmap_mode="r", allow_pickle=False)
    val_images = np.load(data / f"val-{size}.npy", mmap_mode="r", allow_pickle=False)
    captions = np.load(data / "train-qwen.npy", mmap_mode="r") if feature_dim else caption_array(rows, vocabulary)
    val_captions = np.load(data / "val-qwen.npy", mmap_mode="r") if feature_dim else caption_array(val_rows, vocabulary)
    preview_features = np.load(data / "preview-qwen.npy") if feature_dim else None
    alphas = noise_schedule(device)
    directory.mkdir(parents=True, exist_ok=True)
    history = checkpoint["history"] if checkpoint else []
    step = checkpoint["step"] if checkpoint else 0
    initial_step = step
    final_step = min(args.steps, step + args.session_batches) if args.session_batches else args.steps
    best = checkpoint["best_val"] if checkpoint else float("inf")
    elapsed_before = checkpoint.get("total_train_seconds", 0) if checkpoint else 0
    if checkpoint and "random_state" in checkpoint:
        restore_random_state(checkpoint["random_state"], rng)
    batch_log = directory / "batches.jsonl"
    repair_batch_log(batch_log, checkpoint)
    last_loss = checkpoint.get("last_loss") if checkpoint else None
    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        print("Stop requested; saving the current batch before exit.", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    started, report_time, running_loss = time.monotonic(), time.monotonic(), 0.0
    reported_batches = 0

    def snapshot():
        return {
            "format": 1, "task": "scene", "initialization": "random", "model": model.state_dict(),
            "ema": ema.state_dict(), "optimizer": optimizer.state_dict(), "config": config,
            "vocabulary": vocabulary, "data_hash": fingerprint, "size": size, "step": step,
            "history": history, "best_val": best, "arguments": vars(args),
            "total_train_seconds": time.monotonic() - started + elapsed_before,
            "language_features": feature_metadata, "random_state": capture_random_state(rng),
            "last_loss": last_loss,
            "attention_backend": "math", "cudnn_deterministic": True,
        }

    print(json.dumps({"parameters": sum(p.numel() for p in model.parameters()), "images": len(images),
                      "vocabulary": len(vocabulary), "size": size, "start_step": step, "hours_limit": args.hours}), flush=True)
    while (step < final_step and time.monotonic() - started < args.hours * 3600
           and not stop_requested and not (directory / "STOP").exists()):
        model.train()
        pictures, tokens = training_batch(images, captions, args.batch_size, rng, device)
        warmup = min(1.0, (step + 1) / 500)
        for group in optimizer.param_groups:
            group["lr"] = 2e-4 * warmup
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = diffusion_loss(model, pictures, tokens, alphas, drop_condition=True)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite diffusion loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1
        decay = min(0.999, (1 + step) / (10 + step))
        with torch.no_grad():
            for average, current in zip(ema.parameters(), model.parameters()):
                average.lerp_(current, 1 - decay)
        last_loss = float(loss.detach())
        running_loss += last_loss
        reported_batches += 1
        if step % args.checkpoint_every == 0:
            save_checkpoint(directory / "last.pt", snapshot())
        with batch_log.open("a", encoding="utf-8") as output:
            output.write(json.dumps({"step": step, "loss": last_loss, "batch_size": args.batch_size,
                                     "seconds": time.monotonic() - started + elapsed_before}) + "\n")
            output.flush()
        if step % 100 == 0:
            elapsed = time.monotonic() - started
            print(json.dumps({"step": step, "loss": running_loss / reported_batches, "seconds": elapsed,
                              "seconds_per_step": (time.monotonic() - report_time) / reported_batches}), flush=True)
            running_loss, report_time = 0.0, time.monotonic()
            reported_batches = 0
        at_limit = time.monotonic() - started >= args.hours * 3600
        if step % args.save_every == 0 or step == final_step or at_limit:
            val_loss = validation(ema, val_images, val_captions, alphas, device)
            history.append({"step": step, "val_noise_loss": val_loss, "seconds": time.monotonic() - started + elapsed_before})
            improved = val_loss < best
            best = min(best, val_loss)
            record = snapshot()
            save_checkpoint(directory / "last.pt", record)
            if improved:
                save_checkpoint(directory / "best.pt", record)
            save_json(directory / "history.json", history)
            save_preview(ema, vocabulary, directory, step, device, size, preview_features)
            save_checkpoint(directory / "last.pt", snapshot())
            print(json.dumps({"checkpoint_step": step, "val_noise_loss": val_loss}), flush=True)
    # A time limit can be reached between a save and the next loop condition.
    if step > initial_step and (not history or history[-1]["step"] != step):
        val_loss = validation(ema, val_images, val_captions, alphas, device)
        history.append({"step": step, "val_noise_loss": val_loss, "seconds": time.monotonic() - started + elapsed_before})
        improved = val_loss < best
        best = min(best, val_loss)
        record = snapshot()
        save_checkpoint(directory / "last.pt", record)
        if improved:
            save_checkpoint(directory / "best.pt", record)
        save_json(directory / "history.json", history)
        save_preview(ema, vocabulary, directory, step, device, size, preview_features)
        save_checkpoint(directory / "last.pt", snapshot())
    print(f"Training stopped cleanly at step {step}; {(time.monotonic() - started) / 60:.1f} minutes this session.", flush=True)


if __name__ == "__main__":
    main()
