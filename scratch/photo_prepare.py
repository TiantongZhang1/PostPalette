"""Reuse existing COCO subset and official instance annotations; no trained labeler."""

from hashlib import sha256
import json
from zipfile import ZipFile

import numpy as np

from .config import DATA
from .photo_tags import GROUPS, LABELS


def main():
    directory = DATA / "photo_tags"
    directory.mkdir(parents=True, exist_ok=True)
    coco = DATA / "coco"
    rows = {split: json.loads((coco / f"{split}.json").read_text(encoding="utf-8")) for split in ("train", "val", "test")}
    ids = {split: {row["id"]: i for i, row in enumerate(items)} for split, items in rows.items()}
    assert all(not set(ids[a]) & set(ids[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test")))
    targets = {split: np.zeros((len(items), len(LABELS)), dtype=np.float32) for split, items in rows.items()}
    sources = {}
    with ZipFile(coco / "annotations_trainval2017.zip") as archive:
        for official in ("train2017", "val2017"):
            print("Reading official instance annotations: " + official, flush=True)
            raw = archive.read(f"annotations/instances_{official}.json")
            sources[official] = sha256(raw).hexdigest()
            annotations = json.loads(raw)
            del raw
            categories = {category["id"]: category["name"] for category in annotations["categories"]}
            assert set(categories.values()) == set(LABELS), "Taxonomy must cover all official COCO classes"
            mapping = {category_id: [i for i, key in enumerate(LABELS) if name in GROUPS[key][1]] for category_id, name in categories.items()}
            for annotation in annotations["annotations"]:
                # Include all annotated instances, including small/crowd objects.
                # This is presence classification, not localization or activity recognition.
                for split in (("train",) if official == "train2017" else ("val", "test")):
                    index = ids[split].get(annotation["image_id"])
                    if index is not None:
                        targets[split][index, mapping[annotation["category_id"]]] = 1
            del annotations
    manifest = {"dataset": "COCO 2017 official instance categories (80 classes)", "source": "https://cocodataset.org/#download",
                "labels": list(LABELS), "groups": GROUPS, "annotation_sha256": sources,
                "preprocessing": "Reuse aspect-preserving gray-padded RGB 64x64 images; normalize to [0,1]",
                "splits": {}, "warning": "Object presence only, not actions, relationships or scene labels."}
    fingerprint = sha256()
    for split in rows:
        np.save(directory / f"{split}-labels.npy", targets[split])
        source_ids = [row["id"] for row in rows[split]]
        fingerprint.update(json.dumps(source_ids).encode())
        fingerprint.update(targets[split].tobytes())
        fingerprint.update(sha256((coco / f"{split}-64.npy").read_bytes()).digest())
        manifest["splits"][split] = {"images": len(source_ids), "positive_counts": dict(zip(LABELS, targets[split].sum(0).astype(int).tolist()))}
    manifest["data_hash"] = fingerprint.hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest["splits"]), flush=True)


if __name__ == "__main__":
    main()
