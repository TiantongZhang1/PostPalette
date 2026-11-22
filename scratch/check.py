"""Check prepared data and CPU forward passes without training or loading Qwen."""

from hashlib import sha256
import json

import numpy as np
import torch

from .config import ARTIFACTS, CATEGORIES, DATA, EMOTIONS, encode
from .networks import ConditionalVAE, DoodleCNN, EmotionEncoder, vae_loss
from .train import classification_metrics, load_datasets


def check_data():
    text_seen, bitmap_seen = set(), set()
    for split in ("test", "val", "train"):
        rows = json.loads((DATA / "goemotions" / (split + ".json")).read_text(encoding="utf-8"))
        assert rows, f"Empty text split {split}"
        for row in rows:
            key = " ".join(row["text"].lower().split())
            assert key not in text_seen, f"Duplicate text in {split}"
            text_seen.add(key)
            assert len(row["labels"]) == len(EMOTIONS) and any(row["labels"])
            assert all(value in (0, 1) for value in row["labels"])
        with np.load(DATA / "quickdraw" / (split + ".npz"), allow_pickle=False) as data:
            images, labels = data["images"], data["labels"]
        assert images.dtype == np.uint8 and images.shape[1:] == (1, 28, 28)
        assert len(images) == len(labels) and set(labels.tolist()) == set(range(len(CATEGORIES)))
        for image in images:
            digest = sha256(image.tobytes()).digest()
            assert digest not in bitmap_seen, f"Duplicate image in {split}"
            bitmap_seen.add(digest)
        print(f"{split}: {len(rows)} texts, {len(images)} drawings", flush=True)


def main():
    check_data()
    datasets, vocabulary, data_hash = load_datasets("emotion")
    with torch.inference_mode():
        encoder = EmotionEncoder(vocab_size=len(vocabulary)).cpu().eval()
        ids = torch.tensor([encode("Finally done!", vocabulary), encode("123", vocabulary)])
        emotion_logits = encoder(ids)
        assert emotion_logits.shape == (2, 8) and torch.isfinite(emotion_logits).all()
        cnn = DoodleCNN().cpu().eval()
        vae = ConditionalVAE().cpu().eval()
        inputs = torch.zeros(2, 1, 28, 28)
        labels = torch.tensor([0, 5])
        logits, mean, variance = vae(inputs, labels, sample=False)
        loss, _, _ = vae_loss(logits, inputs, mean, variance, 1)
        assert logits.shape == inputs.shape and torch.isfinite(loss)
        generated = vae.decode(torch.zeros(2, vae.latent), labels).sigmoid()
        assert cnn(generated).shape == (2, 6)
        truth = torch.eye(8).bool()
        assert classification_metrics(truth, truth, True)["macro_f1"] == 1
        assert classification_metrics(torch.arange(6), torch.arange(6), False)["accuracy"] == 1
    from models import ModelError, generate_image, parse_suggestions, suggest
    candidates = [{"name": f"template-{number}"} for number in range(3)]
    output = '{"meme_numbers":[3,1,2],"category":"cat","caption":"Finally done!"}'
    ranked, category, _ = parse_suggestions(output, candidates)
    assert ranked[0]["name"] == "template-2" and category == "cat"
    # Missing-checkpoint paths must fail before Qwen or any image weights are loaded.
    if not (ARTIFACTS / "emotion" / "best.pt").exists():
        try:
            suggest("test", True)
        except ModelError:
            pass
        else:
            raise AssertionError("Expected a missing checkpoint error")
    if not (ARTIFACTS / "vae" / "best.pt").exists():
        try:
            generate_image("cat", "#25203c", True, 2026)
        except ModelError:
            pass
        else:
            raise AssertionError("Expected a missing checkpoint error")
    print(f"CPU-only checks passed; emotion data fingerprint {data_hash[:12]}.", flush=True)
    print("No training, optimizer steps, CUDA calls, or Qwen loading were performed.", flush=True)


if __name__ == "__main__":
    main()
