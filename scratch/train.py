"""Train a randomly initialized model, or evaluate its saved checkpoint."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .config import ARTIFACTS, CATEGORIES, DATA, EMOTIONS, encode, tokenize
from .networks import create_network, vae_loss


def save_json(path, value):
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def save_checkpoint(path, value):
    temp = path.with_suffix(".pt.tmp")
    with temp.open("wb") as output:
        torch.save(value, output)
        output.flush()
        os.fsync(output.fileno())
    temp.replace(path)


def fingerprint(directory):
    digest = sha256()
    for path in sorted(directory.iterdir()):
        if path.name != "manifest.json" and path.suffix in (".json", ".npz"):
            digest.update(path.name.encode())
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def load_datasets(task, vocabulary=None):
    if task == "emotion":
        directory = DATA / "goemotions"
        rows = {split: json.loads((directory / (split + ".json")).read_text(encoding="utf-8"))
                for split in ("train", "val", "test")}
        if vocabulary is None:
            counts = Counter(word for row in rows["train"] for word in tokenize(row["text"]))
            words = sorted(counts, key=lambda word: (-counts[word], word))[:11998]
            vocabulary = {"<pad>": 0, "<unk>": 1, **{word: i + 2 for i, word in enumerate(words)}}
        datasets = {
            split: TensorDataset(
                torch.tensor([encode(row["text"], vocabulary) for row in examples], dtype=torch.long),
                torch.tensor([row["labels"] for row in examples], dtype=torch.float32),
            )
            for split, examples in rows.items()
        }
    else:
        directory = DATA / "quickdraw"
        datasets = {}
        for split in ("train", "val", "test"):
            with np.load(directory / (split + ".npz"), allow_pickle=False) as data:
                datasets[split] = TensorDataset(
                    torch.from_numpy(data["images"].astype(np.float32) / 255),
                    torch.from_numpy(data["labels"].copy()).long(),
                )
    return datasets, vocabulary, fingerprint(directory)


def classification_metrics(predicted, actual, multilabel):
    if not multilabel:
        predicted = F.one_hot(predicted, len(CATEGORIES)).bool()
        actual = F.one_hot(actual, len(CATEGORIES)).bool()
    tp = (predicted & actual).sum(0).float()
    fp = (predicted & ~actual).sum(0).float()
    fn = (~predicted & actual).sum(0).float()
    f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
    metrics = {"macro_f1": float(f1.mean()), "per_class_f1": f1.tolist()}
    if multilabel:
        metrics["micro_f1"] = float(2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum()).clamp_min(1))
    else:
        metrics["accuracy"] = float((predicted & actual).sum(1).float().mean())
    return metrics


def calculate_loss(task, model, inputs, labels, beta, pos_weight, sample):
    if task == "vae":
        logits, mean, log_variance = model(inputs, labels, sample=sample)
        loss, reconstruction, divergence = vae_loss(logits, inputs, mean, log_variance, beta)
        return loss, None, {"bce_per_image": float(reconstruction.detach()), "kl_per_image": float(divergence.detach())}
    logits = model(inputs)
    if task == "emotion":
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
    else:
        loss = F.cross_entropy(logits, labels)
    return loss, logits, {}


@torch.inference_mode()
def evaluate(task, model, loader, device, pos_weight):
    model.eval()
    totals = {"loss": 0.0}
    predictions, targets = [], []
    seen = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        loss, logits, details = calculate_loss(task, model, inputs, labels, 1.0, pos_weight, sample=False)
        count = len(inputs)
        seen += count
        totals["loss"] += float(loss) * count
        for name, value in details.items():
            totals[name] = totals.get(name, 0) + value * count
        if logits is not None:
            predictions.append((logits.sigmoid() >= 0.5).cpu() if task == "emotion" else logits.argmax(1).cpu())
            targets.append(labels.bool().cpu() if task == "emotion" else labels.cpu())
    metrics = {name: value / seen for name, value in totals.items()}
    if predictions:
        metrics.update(classification_metrics(torch.cat(predictions), torch.cat(targets), task == "emotion"))
    return metrics


def resolve_device(task, requested):
    """Generator tasks default to CPU; the original three keep their documented CUDA default."""
    if requested is not None:
        return requested
    return "cpu" if task == "gan" else "cuda"


@torch.inference_mode()
def recognizability(images, labels, cnn):
    """Share of drawings the trained classifier assigns to their requested category."""
    return float(cnn(images).argmax(1).eq(labels).float().mean())


def doodle_classifier(device):
    """Load the trained CNN for model selection. It only ever runs forward."""
    path = ARTIFACTS / "cnn" / "best.pt"
    if not path.is_file():
        raise SystemExit("artifacts/cnn/best.pt is missing. Train the CNN before the GAN.")
    record = torch.load(path, map_location="cpu", weights_only=True)
    cnn = create_network("cnn", record["config"]).to(device).eval()
    cnn.load_state_dict(record["model"])
    for parameter in cnn.parameters():
        parameter.requires_grad_(False)
    return cnn


@torch.inference_mode()
def save_samples(model, directory, device, task="vae"):
    from PIL import Image

    model.eval()
    labels = torch.arange(len(CATEGORIES), device=device).repeat_interleave(8)
    noise = torch.randn(len(labels), model.latent, generator=torch.Generator().manual_seed(2026)).to(device)
    samples = (model.generate(noise, labels) if task == "gan"
               else model.decode(noise, labels).sigmoid()).cpu().numpy()
    grid = Image.new("L", (28 * 8, 28 * len(CATEGORIES)), 255)
    for index, drawing in enumerate(samples):
        bitmap = Image.fromarray(np.uint8((1 - drawing[0]) * 255))
        grid.paste(bitmap, ((index % 8) * 28, (index // 8) * 28))
    grid.resize((28 * 8 * 3, 28 * len(CATEGORIES) * 3)).save(directory / "samples.png")


def gan_optimizers(model, lr):
    generator = list(model.condition.parameters()) + list(model.project.parameters()) + list(model.generator.parameters())
    discriminator = list(model.discriminator.parameters()) + list(model.real.parameters()) + list(model.auxiliary.parameters())
    return (torch.optim.AdamW(generator, lr=lr, betas=(0.5, 0.999), weight_decay=0.0001),
            torch.optim.AdamW(discriminator, lr=lr, betas=(0.5, 0.999), weight_decay=0.0001))


@torch.inference_mode()
def validate_gan(model, labels, cnn, device, seed):
    """The GAN loss says nothing about drawing quality, so select on classifier recognizability."""
    model.eval()
    scores, seen = 0.0, 0
    for start in range(0, len(labels), 256):
        batch = labels[start:start + 256].to(device)
        noise = torch.randn(len(batch), model.latent, generator=torch.Generator().manual_seed(seed + start)).to(device)
        scores += recognizability(model.generate(noise, batch), batch, cnn) * len(batch)
        seen += len(batch)
    model.train()
    return {"recognizability": scores / seen}


def train_gan(args, datasets, data_hash, device, directory, checkpoint, epochs, config):
    directory.mkdir(parents=True, exist_ok=True)
    model = create_network("gan", config).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    cnn = doodle_classifier(device)
    optimizers = gan_optimizers(model, args.lr)
    history, updates, start_epoch, best_score = [], 0, 1, float("-inf")
    if checkpoint:
        for optimizer, name in zip(optimizers, ("g", "d")):
            optimizer.load_state_dict(checkpoint["optimizer"][name])
        history, updates = checkpoint["history"], checkpoint["trained_updates"]
        start_epoch, best_score = checkpoint["epoch"] + 1, checkpoint["best_score"]
    validation_labels = datasets["val"].tensors[1]
    print(f"Training gan from scratch on {device}; {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    for epoch in range(start_epoch, epochs + 1):
        torch.manual_seed(args.seed + epoch)
        model.train()
        loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(args.seed + epoch))
        started, total, seen = time.monotonic(), 0.0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            noise = torch.randn(len(images), model.latent, device=device)
            fake = model.generate(noise, labels)

            # Discriminator: real drawings are real and carry their category; generated ones are fake.
            optimizers[1].zero_grad(set_to_none=True)
            real_logit, real_classes = model.discriminate(images)
            fake_logit, fake_classes = model.discriminate(fake.detach())
            # One-sided label smoothing on the real target only.
            discriminator_loss = (
                F.binary_cross_entropy_with_logits(real_logit, torch.full_like(real_logit, 0.9))
                + F.binary_cross_entropy_with_logits(fake_logit, torch.zeros_like(fake_logit))
                + F.cross_entropy(real_classes, labels) + F.cross_entropy(fake_classes, labels)
            )
            discriminator_loss.backward()
            nn.utils.clip_grad_norm_(model.discriminator.parameters(), 1.0)
            optimizers[1].step()

            # Generator: make the discriminator call the drawing real and read the right category.
            optimizers[0].zero_grad(set_to_none=True)
            fake_logit, fake_classes = model.discriminate(fake)
            generator_loss = (F.binary_cross_entropy_with_logits(fake_logit, torch.ones_like(fake_logit))
                              + F.cross_entropy(fake_classes, labels))
            if not torch.isfinite(generator_loss) or not torch.isfinite(discriminator_loss):
                raise RuntimeError("Non-finite loss; no checkpoint was saved for this epoch")
            generator_loss.backward()
            nn.utils.clip_grad_norm_(model.generator.parameters(), 1.0)
            optimizers[0].step()

            total += float(generator_loss.detach()) * len(images)
            seen += len(images)
            updates += 1
        metrics = validate_gan(model, validation_labels, cnn, device, args.seed + epoch)
        score = metrics["recognizability"]
        improved = score > best_score
        best_score = max(best_score, score)
        history.append({"epoch": epoch, "train_loss": total / seen, "val": metrics,
                        "beta": None, "seconds": time.monotonic() - started})
        checkpoint = {
            "format": 1, "task": "gan", "config": config, "model": model.state_dict(),
            "optimizer": {"g": optimizers[0].state_dict(), "d": optimizers[1].state_dict()},
            "epoch": epoch, "trained_updates": updates, "initialization": "random",
            "data_hash": data_hash, "vocabulary": None, "labels": list(CATEGORIES),
            "arguments": vars(args), "history": history, "best_score": best_score,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        if improved:
            save_checkpoint(directory / "best.pt", checkpoint)
            save_samples(model, directory, device, task="gan")
        save_checkpoint(directory / "last.pt", checkpoint)
        save_json(directory / "history.json", history)
        print(json.dumps(history[-1]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("emotion", "cnn", "vae", "gan"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None,
                        help="Defaults to cuda, except gan which defaults to cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the best checkpoint on the test split without training")
    args = parser.parse_args()
    epochs = args.epochs if args.epochs is not None else {"emotion": 15, "cnn": 12, "vae": 40, "gan": 40}[args.task]
    if epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch size, and learning rate must be positive")
    if args.resume and args.evaluate:
        parser.error("Choose --resume or --evaluate")
    if args.evaluate and args.task == "gan":
        parser.error("A generator has no meaningful test-split loss. Inspect artifacts/gan/samples.png "
                     "and the validation recognizability in artifacts/gan/history.json instead.")
    device_name = resolve_device(args.task, args.device)
    directory = ARTIFACTS / args.task
    checkpoint_path = directory / ("best.pt" if args.evaluate else "last.pt")
    checkpoint = None
    if args.resume or args.evaluate:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    elif (directory / "last.pt").exists():
        parser.error("A run already exists. Use --resume to continue it, or move its artifact folder before starting another run.")
    try:
        datasets, vocabulary, data_hash = load_datasets(args.task, checkpoint.get("vocabulary") if checkpoint else None)
    except FileNotFoundError:
        parser.error("Prepared data is missing. Run python -m scratch.prepare first.")
    if checkpoint and (checkpoint["data_hash"] != data_hash or checkpoint["task"] != args.task):
        parser.error("The checkpoint does not match this dataset/task")
    if args.resume:
        original = checkpoint["arguments"]
        if any(original[name] != getattr(args, name) for name in ("seed", "lr", "batch_size")):
            parser.error("Resume must use the original seed, learning rate, and batch size")
        if epochs <= checkpoint["epoch"]:
            parser.error("--epochs must exceed the last completed epoch")
    # No CUDA call or allocation happens until an explicit train/evaluate command reaches this point.
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Install CUDA PyTorch or pass --device cpu.")
    device = torch.device(device_name)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = {"vocab_size": len(vocabulary)} if args.task == "emotion" else {}
    if checkpoint:
        config = checkpoint["config"]
    if args.task == "gan":
        train_gan(args, datasets, data_hash, device, directory, checkpoint, epochs, config)
        return
    model = create_network(args.task, config).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    pos_weight = None
    if args.task == "emotion":
        positives = datasets["train"].tensors[1].sum(0)
        pos_weight = ((len(datasets["train"]) - positives) / positives.clamp_min(1)).clamp(1, 5).to(device)
    if args.evaluate:
        metrics = evaluate(args.task, model, DataLoader(datasets["test"], batch_size=args.batch_size), device, pos_weight)
        report = {"split": "test", "checkpoint_epoch": checkpoint["epoch"], "data_hash": data_hash,
                  "labels": EMOTIONS if args.task == "emotion" else CATEGORIES, "metrics": metrics}
        save_json(directory / "test_metrics.json", report)
        print(json.dumps(report, indent=2), flush=True)
        return

    directory.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01 if args.task == "emotion" else 0.0001)
    history, updates, start_epoch, best_score = [], 0, 1, float("-inf")
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        history, updates = checkpoint["history"], checkpoint["trained_updates"]
        start_epoch, best_score = checkpoint["epoch"] + 1, checkpoint["best_score"]
    print(f"Training {args.task} from scratch on {device}; {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    for epoch in range(start_epoch, epochs + 1):
        torch.manual_seed(args.seed + epoch)
        model.train()
        loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(args.seed + epoch))
        started, total, seen = time.monotonic(), 0.0, 0
        beta = min(1.0, epoch / 10)
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = calculate_loss(args.task, model, inputs, labels, beta, pos_weight, sample=True)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; no checkpoint was saved for this epoch")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(inputs)
            seen += len(inputs)
            updates += 1
        metrics = evaluate(args.task, model, DataLoader(datasets["val"], batch_size=args.batch_size), device, pos_weight)
        score = metrics.get("macro_f1", -metrics["loss"])
        improved = score > best_score
        best_score = max(best_score, score)
        history.append({"epoch": epoch, "train_loss": total / seen, "val": metrics,
                        "beta": beta if args.task == "vae" else None, "seconds": time.monotonic() - started})
        checkpoint = {
            "format": 1, "task": args.task, "config": config, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "epoch": epoch, "trained_updates": updates,
            "initialization": "random", "data_hash": data_hash, "vocabulary": vocabulary,
            "labels": list(EMOTIONS if args.task == "emotion" else CATEGORIES),
            "arguments": vars(args), "history": history, "best_score": best_score,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        if improved:
            save_checkpoint(directory / "best.pt", checkpoint)
            if args.task == "vae":
                save_samples(model, directory, device)
        save_checkpoint(directory / "last.pt", checkpoint)
        save_json(directory / "history.json", history)
        print(json.dumps(history[-1]), flush=True)


if __name__ == "__main__":
    main()
