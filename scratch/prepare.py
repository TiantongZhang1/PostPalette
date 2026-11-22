"""Download public data and make reproducible splits. No model is loaded."""

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np

from .config import CATEGORIES, DATA, EMOTIONS, EMOTION_GROUPS

GOEMOTIONS = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/"
QUICKDRAW = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"


def read_url(url):
    return urlopen(Request(url, headers={"User-Agent": "PostPalette/1.0"}), timeout=45)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_emotions(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with read_url(GOEMOTIONS + "emotions.txt") as response:
        names = response.read().decode("utf-8").splitlines()
    if set(names) != set(EMOTION_GROUPS):
        raise ValueError("GoEmotions taxonomy changed")
    seen_texts = set()
    sources, counts = {}, {}
    # Give the official test/dev sets priority when removing exact text duplicates.
    for split, filename in (("test", "test.tsv"), ("val", "dev.tsv"), ("train", "train.tsv")):
        url = GOEMOTIONS + filename
        with read_url(url) as response:
            raw = response.read()
        (directory / filename).write_bytes(raw)
        rows = []
        for text, labels, example_id in csv.reader(StringIO(raw.decode("utf-8")), delimiter="\t"):
            normalized = " ".join(text.lower().split())
            if not normalized or normalized in seen_texts:
                continue
            seen_texts.add(normalized)
            groups = {EMOTION_GROUPS[names[int(label)]] for label in labels.split(",")}
            rows.append({"id": example_id, "text": text, "labels": [int(group in groups) for group in EMOTIONS]})
        write_json(directory / (split + ".json"), rows)
        sources[split] = {"url": url, "raw_sha256": sha256(raw).hexdigest()}
        counts[split] = len(rows)
        print(f"GoEmotions {split}: {len(rows)} rows", flush=True)
    write_json(directory / "manifest.json", {
        "dataset": "GoEmotions", "labels": EMOTIONS, "mapping": EMOTION_GROUPS,
        "splits": counts, "sources": sources, "license": "Apache-2.0",
        "deduplication": "Exact lowercased whitespace-normalized text; test then val then train priority",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    })


def read_exact(response, size):
    chunks = bytearray()
    while len(chunks) < size:
        chunk = response.read(min(1024 * 1024, size - len(chunks)))
        if not chunk:
            raise ValueError("Dataset download ended early")
        chunks.extend(chunk)
    return chunks


def prepare_drawings(directory, per_class, seed):
    directory.mkdir(parents=True, exist_ok=True)
    all_images, all_labels, all_splits, sources = [], [], [], []
    seen = set()
    rng = np.random.default_rng(seed)
    for label, category in enumerate(CATEGORIES):
        url = QUICKDRAW + quote(category) + ".npy"
        # Stream only the first N rows to keep preparation small and CPU-only.
        # The manifest explicitly records this prefix sampling.
        with read_url(url) as response:
            version = np.lib.format.read_magic(response)
            if version != (1, 0):
                raise ValueError(f"Unexpected NPY header version {version}")
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(response)
            if fortran_order or dtype != np.dtype("uint8") or len(shape) != 2 or shape[1] != 784:
                raise ValueError("Expected row-major uint8 QuickDraw bitmaps")
            count = min(per_class, shape[0])
            raw = read_exact(response, count * 784)
        images = np.frombuffer(raw, dtype=np.uint8).reshape(count, 1, 28, 28)
        keep = []
        for index, image in enumerate(images):
            digest = sha256(image.tobytes()).digest()
            if digest not in seen and np.count_nonzero(image) >= 8:
                seen.add(digest)
                keep.append(index)
        images = images[keep]
        if len(images) < 100:
            raise ValueError(f"Too few unique images in {category}")
        permutation = rng.permutation(len(images))
        images = images[permutation]
        split_ids = np.zeros(len(images), dtype=np.uint8)
        split_ids[int(len(images) * 0.8):int(len(images) * 0.9)] = 1
        split_ids[int(len(images) * 0.9):] = 2
        all_images.append(images)
        all_labels.append(np.full(len(images), label, dtype=np.int64))
        all_splits.append(split_ids)
        sources.append({"category": category, "url": url, "downloaded_rows": count,
                        "unique_kept": len(images), "prefix_sha256": sha256(raw).hexdigest()})
        print(f"QuickDraw {category}: {len(images)} unique drawings", flush=True)
    images, labels, splits = map(np.concatenate, (all_images, all_labels, all_splits))
    counts = {}
    for number, split in enumerate(("train", "val", "test")):
        mask = splits == number
        np.savez_compressed(directory / (split + ".npz"), images=images[mask], labels=labels[mask])
        counts[split] = int(mask.sum())
    write_json(directory / "manifest.json", {
        "dataset": "Quick, Draw!", "categories": CATEGORIES, "splits": counts, "seed": seed,
        "sampling": "First N rows per category, remove exact bitmap duplicates globally, shuffle per category, 80/10/10 split",
        "sources": sources, "license": "CC BY 4.0",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", "goemotions", "quickdraw"), default="all")
    parser.add_argument("--per-class", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.per_class < 200:
        parser.error("--per-class must be at least 200")
    if args.dataset in ("all", "goemotions"):
        prepare_emotions(DATA / "goemotions")
    if args.dataset in ("all", "quickdraw"):
        prepare_drawings(DATA / "quickdraw", args.per_class, args.seed)


if __name__ == "__main__":
    main()
