"""Cache language-only Qwen features; Qwen is the permitted pretrained LLM."""

import argparse
from hashlib import sha256
import json
import os

import numpy as np
import torch
from torch.nn import functional as F

from ..config import DATA, ROOT
from ..train import save_json

QWEN = "Qwen/Qwen2.5-1.5B-Instruct"


@torch.inference_mode()
def text_features(captions, tokenizer, llm):
    device = next(llm.parameters()).device
    batch = tokenizer(captions, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
    hidden = llm.model(**batch, use_cache=False, return_dict=True).last_hidden_state.float()
    features = []
    for row, mask in zip(hidden, batch.attention_mask):
        row = row[mask.bool()]
        features.append(F.adaptive_avg_pool1d(row.transpose(0, 1).unsqueeze(0), 8)[0].transpose(0, 1))
    return torch.stack(features)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    os.environ.setdefault("HF_HOME", str(ROOT / ".model-cache" / "huggingface"))
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN, dtype=torch.float16, local_files_only=True, use_safetensors=True,
    ).cuda().eval()
    directory = DATA / "coco"
    manifests = {}
    for split in ("train", "val", "test"):
        raw = (directory / (split + ".json")).read_bytes()
        rows = json.loads(raw)
        if any(not row["captions"] for row in rows):
            raise ValueError("An image has no human caption")
        digest = sha256(raw).hexdigest()
        path = directory / f"{split}-qwen.npy"
        marker = path.with_suffix(".json")
        if path.exists() and marker.exists() and json.loads(marker.read_text())["captions_sha256"] == digest:
            print(f"{split}: valid cached Qwen features", flush=True)
        else:
            captions = [row["captions"][min(index, len(row["captions"]) - 1)] for row in rows for index in range(5)]
            temporary = path.with_suffix(".partial.npy")
            array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16,
                                             shape=(len(rows), 5, 8, model.config.hidden_size))
            flat = array.reshape(-1, 8, model.config.hidden_size)
            for start in range(0, len(captions), args.batch_size):
                flat[start:start + args.batch_size] = text_features(
                    captions[start:start + args.batch_size], tokenizer, model,
                ).cpu().numpy().astype(np.float16)
                if start % (args.batch_size * 20) == 0:
                    print(f"{split}: {min(start + args.batch_size, len(captions))}/{len(captions)} language captions", flush=True)
            array.flush()
            del flat, array
            temporary.replace(path)
            save_json(marker, {"model": QWEN, "captions_sha256": digest, "tokens": 8,
                               "width": model.config.hidden_size, "pretrained_component": "LLM only",
                               "caption_selection": "First five; repeat last if fewer than five"})
        manifests[split] = json.loads(marker.read_text())
    save_json(directory / "qwen_features.json", manifests)
    from .coco_train import PROMPTS
    np.save(directory / "preview-qwen.npy", text_features(list(PROMPTS), tokenizer, model).cpu().numpy().astype(np.float16))
    print("Qwen language feature preparation complete.", flush=True)


if __name__ == "__main__":
    main()
