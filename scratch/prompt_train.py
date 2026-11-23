"""Train/evaluate a scratch caption-denoising prompt writer on existing COCO captions.

CPU only by design: no CUDA initialization, visual backbone or pretrained embeddings.
"""

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
import random
import time

import torch
from torch.nn import functional as F

from .config import ARTIFACTS, DATA
from .prompt_model import BOS, EOS, PAD, PromptWriter, encode_prompt, words
from .train import save_checkpoint, save_json


def load_data():
    paths = {split: DATA / "coco" / f"{split}.json" for split in ("train", "val", "test")}
    records = {split: json.loads(path.read_text(encoding="utf-8")) for split, path in paths.items()}
    groups = [set(row["id"] for row in records[split]) for split in paths]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("COCO image IDs overlap across splits")
    captions = {split: [caption for row in rows for caption in row["captions"] if 4 <= len(words(caption)) < 40]
                for split, rows in records.items()}
    digest = sha256(b"".join(paths[split].read_bytes() for split in paths)).hexdigest()
    return captions, digest, {split: len(rows) for split, rows in records.items()}


def make_batch(examples, vocabulary, rng):
    # A visual caption is its own target. Training inputs lose 0-20% of words and
    # sometimes acquire a casual prefix; no LLM-generated labels are involved.
    sources = []
    targets = []
    prefixes = ("", "", "today i saw ", "my photo shows ", "look at ")
    for text in examples:
        tokens = words(text)
        drop = rng.uniform(0, 0.2)
        kept = [word for word in tokens if rng.random() >= drop] or tokens[:1]
        sources.append(encode_prompt(rng.choice(prefixes) + " ".join(kept), vocabulary))
        targets.append(encode_prompt(text, vocabulary))
    return torch.tensor(sources), torch.tensor(targets)


def objective(model, source, target):
    inputs = torch.cat((torch.full((len(target), 1), BOS), target[:, :-1]), 1)
    output = model(source, inputs)
    return F.nll_loss(output.flatten(0, 1), target.flatten(), ignore_index=PAD)


@torch.inference_mode()
def evaluate(model, captions, vocabulary, limit=512):
    model.eval()
    rng = random.Random(114)
    numerator = denominator = 0
    for offset in range(0, min(limit, len(captions)), 32):
        source, target = make_batch(captions[offset: min(offset + 32, limit)], vocabulary, rng)
        count = int(target.ne(PAD).sum())
        numerator += float(objective(model, source, target)) * count
        denominator += count
    model.train()
    nll = numerator / denominator
    return {"token_nll": nll, "perplexity": math.exp(min(nll, 20)), "examples": min(limit, len(captions))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--minutes", type=float, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.minutes, args.batch_size, args.threads) <= 0:
        parser.error("Training limits must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(2026)
    rng = random.Random(2026)
    captions, digest, image_counts = load_data()
    directory = ARTIFACTS / "prompt"
    directory.mkdir(parents=True, exist_ok=True)
    record = None
    if args.resume or args.evaluate:
        record = torch.load(directory / ("best.pt" if args.evaluate else "last.pt"), map_location="cpu", weights_only=True)
        if record["data_hash"] != digest:
            raise ValueError("Dataset changed; do not resume/evaluate this checkpoint")
        vocabulary = record["vocabulary"]
        config = record["config"]
    else:
        counts = Counter(word for text in captions["train"] for word in words(text))
        vocabulary = {"<pad>": PAD, "<unk>": 1, "<bos>": BOS, "<eos>": EOS}
        vocabulary.update({word: i + 4 for i, (word, _) in enumerate(counts.most_common(4092))})
        config = {"vocabulary_size": len(vocabulary), "embedding": 64, "hidden": 96}
    model = PromptWriter(**config).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    step, best, history = 0, float("inf"), []
    if record:
        model.load_state_dict(record["model"])
        optimizer.load_state_dict(record["optimizer"])
        step, best, history = record["step"], record["best_val"], record["history"]
        rng.setstate(record["rng"])
        torch.set_rng_state(record["torch_rng"])
    if args.evaluate:
        metrics = evaluate(model, captions["test"], vocabulary, len(captions["test"]))
        probes = ["a group of friends cooking food on a grill in a park",
                  "a cat sleeping on a sofa", "people walking on a beach near the ocean"]
        model.eval()
        metrics.update({"step": step, "split": "test", "image_counts": image_counts,
                        "warning": "Denoising perplexity does not measure image quality or social-post generalization.",
                        "probes": [{"input": text, "output": model.write(text, vocabulary)} for text in probes]})
        save_json(directory / "test_metrics.json", metrics)
        print(json.dumps(metrics), flush=True)
        return
    started = time.monotonic()
    interrupted = False
    print(f"CPU training: {len(captions['train'])} captions, {sum(p.numel() for p in model.parameters())} parameters; resume step {step}", flush=True)
    try:
        while step < args.steps and time.monotonic() - started < args.minutes * 60:
            if (directory / "STOP").exists():
                break
            examples = [rng.choice(captions["train"]) for _ in range(args.batch_size)]
            source, target = make_batch(examples, vocabulary, rng)
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model, source, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            step += 1
            improved = False
            if step % 100 == 0 or step == 1:
                validation = evaluate(model, captions["val"], vocabulary)
                improved = validation["token_nll"] < best
                best = min(best, validation["token_nll"])
                history.append({"step": step, "loss": float(loss.detach()), "val": validation})
                print(json.dumps(history[-1]), flush=True)
            record = {"task": "prompt", "initialization": "random", "step": step,
                      "config": config, "vocabulary": vocabulary, "model": model.state_dict(),
                      "optimizer": optimizer.state_dict(), "rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
                      "data_hash": digest, "best_val": best, "history": history,
                      "dataset": "COCO 2017 caption denoising", "image_counts": image_counts,
                      "caption_counts": {split: len(rows) for split, rows in captions.items()}}
            save_checkpoint(directory / "last.pt", record)
            if improved:
                save_checkpoint(directory / "best.pt", record)
    except KeyboardInterrupt:
        interrupted = True
    save_json(directory / "history.json", history)
    print(f"CPU training {'interrupted' if interrupted else 'stopped'}; saved step {record['step'] if record else 0}. Best checkpoint chosen by validation NLL.", flush=True)


if __name__ == "__main__":
    main()
