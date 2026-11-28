"""Scratch multilabel CNN training, per-batch resume, held-out evaluation."""

import argparse
import copy
import json
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .config import ARTIFACTS, DATA
from .photo_tags import LABELS, PhotoTagCNN
from .train import save_checkpoint, save_json


def scores(probabilities, target, thresholds):
    prediction = probabilities >= np.asarray(thresholds)
    truth = target > 0.5
    tp = (prediction & truth).sum(0)
    fp = (prediction & ~truth).sum(0)
    fn = (~prediction & truth).sum(0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return {"macro_f1": float(f1.mean()), "micro_f1": float(2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1)),
            "per_label": {key: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]),
                                "positives": int(truth[:, i].sum()), "threshold": float(thresholds[i])} for i, key in enumerate(LABELS)}}


@torch.inference_mode()
def predict(model, images):
    model.eval()
    device = next(model.parameters()).device
    outputs = []
    for offset in range(0, len(images), 64):
        batch = torch.from_numpy(np.array(images[offset:offset + 64], copy=True)).to(device).float() / 255
        outputs.append(model(batch).sigmoid().cpu().numpy())
    model.train()
    return np.concatenate(outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--minutes", type=float, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--run", default="photo_tags")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--balanced", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    if min(args.steps, args.minutes, args.batch_size, args.threads) <= 0:
        parser.error("Training limits must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(2026)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA is not available")
        # This small model can use native kernels; avoid the previously unstable cuDNN path.
        torch.backends.cudnn.enabled = False
    rng = random.Random(2026)
    manifest = json.loads((DATA / "photo_tags/manifest.json").read_text(encoding="utf-8"))
    if manifest["labels"] != list(LABELS):
        raise ValueError("Labels changed; prepare data again")
    if not args.run.replace('_', '').isalnum():
        parser.error("Run name must be alphanumeric/underscore")
    directory = ARTIFACTS / args.run
    directory.mkdir(parents=True, exist_ok=True)
    images = {split: np.load(DATA / f"coco/{split}-64.npy", mmap_mode="r") for split in ("train", "val", "test")}
    labels = {split: np.load(DATA / f"photo_tags/{split}-labels.npy") for split in images}
    for split in images:
        assert len(images[split]) == len(labels[split])
    record = None
    config = {"classes": len(LABELS), "width": 16}
    if args.resume or args.evaluate:
        record = torch.load(directory / ("best.pt" if args.evaluate else "last.pt"), map_location="cpu", weights_only=True)
        if record["data_hash"] != manifest["data_hash"]:
            raise ValueError("Dataset changed")
        config = record["config"]
    for option in ('balanced', 'ema'):
        if getattr(args, option) is None:
            setattr(args, option, record.get('training_options', {}).get(option, False) if record else False)
    model = PhotoTagCNN(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate or 0.001, weight_decay=0.0001)
    step, best, history, elapsed_before = 0, -1, [], 0
    if record:
        model.load_state_dict(record.get("train_model", record["model"]))
        optimizer.load_state_dict(record["optimizer"])
        rng.setstate(record["rng"])
        torch.set_rng_state(record["torch_rng"])
        if device.type == "cuda" and record.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(record["cuda_rng"])
        step, best, history = record["step"], record["best_val"], record["history"]
        elapsed_before = record.get("seconds", 0)
        if args.learning_rate is not None:
            for group in optimizer.param_groups:
                group['lr'] = args.learning_rate
    averaged = copy.deepcopy(model).eval()
    if record:
        averaged.load_state_dict(record['model'])
    if args.evaluate:
        probabilities = predict(averaged, images["test"])
        result = scores(probabilities, labels["test"], record["thresholds"])
        result.update({"split": "test", "step": step, "images": len(images["test"]),
                       "warning": "Object presence on COCO, not activities, caption correctness or user-photo accuracy."})
        save_json(directory / "test_metrics.json", result)
        print(json.dumps(result), flush=True)
        return
    positive = labels["train"].sum(0)
    label_indices = [np.flatnonzero(labels['train'][:, i]).tolist() for i in range(len(LABELS))]
    pos_weight = torch.tensor(np.sqrt((len(labels["train"]) - positive) / np.maximum(positive, 1)).clip(1, 6), dtype=torch.float32, device=device)
    thresholds = record["thresholds"] if record else [0.5] * len(LABELS)
    started = time.monotonic()
    print(f"{device} scratch photo CNN, {sum(p.numel() for p in model.parameters())} parameters; starting step {step}", flush=True)
    try:
        while step < args.steps and time.monotonic() - started < args.minutes * 60:
            if (directory / "STOP").exists():
                break
            indices = []
            for _ in range(args.batch_size):
                pool = label_indices[rng.randrange(len(LABELS))]
                indices.append(rng.choice(pool) if args.balanced and rng.random() < 0.5 and pool else rng.randrange(len(images['train'])))
            batch = torch.from_numpy(np.array(images["train"][indices], copy=True)).to(device).float() / 255
            # Entire-photo horizontal flip; no crop that could remove positive objects.
            flip = torch.rand(len(batch), device=device) < 0.5
            batch[flip] = batch[flip].flip(-1)
            target = torch.from_numpy(labels["train"][indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(batch), target, pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            if args.ema:
                with torch.no_grad():
                    for target_param, param in zip(averaged.parameters(), model.parameters()):
                        target_param.lerp_(param, 0.01)
                    for target_buffer, buffer in zip(averaged.buffers(), model.buffers()):
                        if target_buffer.is_floating_point():
                            target_buffer.lerp_(buffer, 0.01)
                        else:
                            target_buffer.copy_(buffer)
            else:
                averaged.load_state_dict(model.state_dict())
            step += 1
            improved = False
            if step % 100 == 0 or step == 1:
                probabilities = predict(averaged, images["val"])
                thresholds = []
                for i in range(len(LABELS)):
                    if labels["val"][:, i].sum() == 0:
                        thresholds.append(0.95)
                        continue
                    choices = np.arange(0.2, 0.81, 0.05)
                    predicted = probabilities[:, i, None] >= choices
                    truth = labels["val"][:, i, None] > 0.5
                    tp = (predicted & truth).sum(0)
                    fp = (predicted & ~truth).sum(0)
                    fn = (~predicted & truth).sum(0)
                    quality = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
                    thresholds.append(float(choices[int(np.argmax(quality))]))
                validation = scores(probabilities, labels["val"], thresholds)
                improved = validation["macro_f1"] > best
                best = max(best, validation["macro_f1"])
                history.append({"step": step, "loss": float(loss.detach()), "val_macro_f1": validation["macro_f1"], "val_micro_f1": validation["micro_f1"]})
                print(json.dumps(history[-1]), flush=True)
            record = {"task": "photo_tags", "initialization": "random", "step": step,
                      "config": config, "labels": list(LABELS), "model": averaged.state_dict(), "train_model": model.state_dict(),
                      "training_options": {"balanced": args.balanced, "ema": args.ema, "learning_rate": optimizer.param_groups[0]['lr']},
                      "optimizer": optimizer.state_dict(), "rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
                      "cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None,
                      "thresholds": thresholds, "best_val": best, "history": history,
                      "data_hash": manifest["data_hash"], "seconds": elapsed_before + time.monotonic() - started,
                      "dataset": "COCO official instance presence (80 classes)", "manifest": manifest, "device": str(device)}
            save_checkpoint(directory / "last.pt", record)
            if improved:
                save_checkpoint(directory / "best.pt", record)
    except KeyboardInterrupt:
        pass
    save_json(directory / "history.json", history)
    print(f"Stopped; last saved step {step}. Validation-selected best.pt is used for inference.", flush=True)


if __name__ == "__main__":
    main()
