"""Prepare a reproducible COCO caption/photo subset without pretrained processing."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from io import BytesIO
import json
import random
import time
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageOps

from .config import DATA
from .prepare import write_json

BASE = "https://s3.amazonaws.com/images.cocodataset.org/"


def download_image(row, directory, size):
    image_path = directory / (str(row["id"]) + ".png")
    metadata_path = image_path.with_suffix(".json")
    if image_path.exists() and metadata_path.exists():
        with Image.open(image_path) as image:
            if image.size == (size, size):
                return row | json.loads(metadata_path.read_text(encoding="utf-8"))
    url = BASE + row["source_split"] + "/" + row["filename"]
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={"User-Agent": "PostPalette/1.0"}), timeout=25) as response:
                content = response.read(20 * 1024 * 1024)
            with Image.open(BytesIO(content)) as original:
                image = ImageOps.pad(ImageOps.exif_transpose(original).convert("RGB"), (size, size),
                                     method=Image.Resampling.LANCZOS, color=(127, 127, 127))
            temp = image_path.with_suffix(".partial")
            image.save(temp, format="PNG")
            temp.replace(image_path)
            metadata = {"source_url": url, "raw_sha256": sha256(content).hexdigest()}
            write_json(metadata_path, metadata)
            return row | metadata
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(0.5 * (attempt + 1))


def read_rows(archive, split):
    content = archive.read(f"annotations/captions_{split}.json")
    data = json.loads(content)
    captions = {}
    for annotation in data["annotations"]:
        captions.setdefault(annotation["image_id"], []).append(annotation["caption"])
    rows = [
        {"id": image["id"], "filename": image["file_name"], "source_split": split,
         "captions": captions[image["id"]], "license": image["license"]}
        for image in data["images"] if image["id"] in captions
    ]
    return rows, sha256(content).hexdigest(), data["licenses"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-images", type=int, default=20000)
    parser.add_argument("--val-images", type=int, default=500)
    parser.add_argument("--test-images", type=int, default=500)
    parser.add_argument("--size", type=int, choices=(64, 128), default=64)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if min(args.train_images, args.val_images, args.test_images, args.workers) < 1:
        parser.error("Image counts and workers must be positive")
    directory = DATA / "coco"
    images_directory = directory / f"images-{args.size}"
    images_directory.mkdir(parents=True, exist_ok=True)
    with ZipFile(directory / "annotations_trainval2017.zip") as archive:
        train, train_hash, licenses = read_rows(archive, "train2017")
        heldout, heldout_hash, _ = read_rows(archive, "val2017")
    if args.train_images > len(train) or args.val_images + args.test_images > len(heldout):
        parser.error("Requested subset exceeds official split sizes")
    rng = random.Random(args.seed)
    rng.shuffle(train)
    rng.shuffle(heldout)
    splits = {
        "train": train[:args.train_images],
        "val": heldout[:args.val_images],
        "test": heldout[args.val_images:args.val_images + args.test_images],
    }
    tasks = [(split, row) for split, rows in splits.items() for row in rows]
    results, failures = {split: {} for split in splits}, []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_image, row, images_directory, args.size): (split, row) for split, row in tasks}
        for number, future in enumerate(as_completed(futures), 1):
            split, row = futures[future]
            try:
                result = future.result()
                results[split][row["id"]] = result
            except Exception as error:
                failures.append({"id": row["id"], "error": str(error)})
            if number % 250 == 0 or number == len(tasks):
                elapsed = time.monotonic() - started
                print(f"{number}/{len(tasks)} images, {len(failures)} failed, {number / max(elapsed, 1):.1f}/s", flush=True)
    write_json(directory / "download_failures.json", failures)
    if len(failures) > len(tasks) * 0.05:
        raise RuntimeError("More than 5% downloads failed; rerun preparation to reuse the cache")
    seen, counts = set(), {}
    for split in ("test", "val", "train"):
        rows = []
        for original in splits[split]:
            row = results[split].get(original["id"])
            if row is None:
                continue
            with Image.open(images_directory / (str(row["id"]) + ".png")) as image:
                digest = sha256(image.tobytes()).hexdigest()
            if digest not in seen:
                seen.add(digest)
                rows.append(row)
        if not rows:
            raise RuntimeError(f"No usable {split} images")
        array = np.lib.format.open_memmap(directory / f"{split}-{args.size}.npy", mode="w+",
                                          dtype=np.uint8, shape=(len(rows), 3, args.size, args.size))
        for index, row in enumerate(rows):
            with Image.open(images_directory / (str(row["id"]) + ".png")) as image:
                array[index] = np.asarray(image).transpose(2, 0, 1)
        array.flush()
        del array
        write_json(directory / f"{split}.json", rows)
        counts[split] = len(rows)
    write_json(directory / "manifest.json", {
        "dataset": "COCO 2017 captions", "size": args.size, "splits": counts, "seed": args.seed,
        "annotation_sha256": {"train2017": train_hash, "val2017": heldout_hash},
        "selection": "Seeded random subset of official train2017; disjoint val/test subsets of official val2017",
        "preprocessing": "EXIF transpose, RGB, aspect-preserving resize with gray padding; no trained processor",
        "deduplication": "Exact resized RGB bitmap hash; held-out splits take priority",
        "licenses": licenses, "source": "https://cocodataset.org/#download",
        "failed_downloads": len(failures),
    })
    print(json.dumps({"splits": counts, "size": args.size}), flush=True)


if __name__ == "__main__":
    main()
