"""Held-out denoising and fixed-noise text sensitivity checks, using no pretrained evaluators."""

import json

import numpy as np
from PIL import Image
import torch

from .config import ARTIFACTS, DATA
from .coco_train import PROMPTS, caption_array, encode_caption, diffusion_loss, dataset_hash
from .photo_diffusion import PhotoUNet, configure_cuda, noise_schedule, sample
from .train import save_json


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        configure_cuda()
    torch.set_num_threads(4)
    directory = ARTIFACTS / "scene"
    record = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    if dataset_hash(DATA / "coco", record["size"]) != record["data_hash"]:
        raise ValueError("Test data changed since training")
    model = PhotoUNet(**record["config"]).to(device).eval()
    model.load_state_dict(record["ema"])
    images = np.load(DATA / "coco" / f"test-{record['size']}.npy", mmap_mode="r")
    rows = json.loads((DATA / "coco" / "test.json").read_text(encoding="utf-8"))
    uses_qwen = bool(record["config"].get("feature_dim"))
    captions = np.load(DATA / "coco" / "test-qwen.npy", mmap_mode="r") if uses_qwen else caption_array(rows, record["vocabulary"])
    alphas = noise_schedule(device)
    scores = {}
    with torch.inference_mode():
        for condition in ("paired", "shuffled"):
            generator = torch.Generator(device=device).manual_seed(9026)
            total, count = 0.0, 0
            for start in range(0, len(images), 16):
                end = min(start + 16, len(images))
                photos = torch.from_numpy(images[start:end].copy()).to(device, dtype=torch.float32).div_(127.5).sub_(1)
                indices = range(start, end) if condition == "paired" else ((i + 137) % len(images) for i in range(start, end))
                tokens = torch.as_tensor(captions[list(indices), 0].copy(), device=device) if uses_qwen else torch.stack([captions[i][0] for i in indices]).to(device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    loss = diffusion_loss(model, photos, tokens, alphas, generator=generator)
                total += float(loss) * len(photos)
                count += len(photos)
            scores[condition + "_noise_loss"] = total / count
        tokens = torch.as_tensor(np.load(DATA / "coco" / "preview-qwen.npy"), device=device) if uses_qwen else torch.tensor([encode_caption(prompt, record["vocabulary"]) for prompt in PROMPTS], device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = sample(model, tokens, size=record["size"], steps=1000, seed=42, same_noise=True).float().cpu()
        native = record["size"]
        grid = Image.new("RGB", (native * len(PROMPTS), native))
        for index, image in enumerate(outputs):
            grid.paste(Image.fromarray(np.uint8(image.permute(1, 2, 0).numpy() * 255)), (index * native, 0))
        grid.resize((native * len(PROMPTS) * 4, native * 4), Image.Resampling.NEAREST).save(directory / "test-prompts.png")
        scores["mean_prompt_pixel_difference_same_noise"] = float((outputs[1:] - outputs[:1]).abs().mean())
    report = {
        "split": "test", "images": len(images), "checkpoint_step": record["step"], "data_hash": record["data_hash"],
        "metrics": scores,
        "interpretation": "Denoising loss and prompt sensitivity do not establish realism or semantic correctness.",
        "prompts_left_to_right": list(PROMPTS),
    }
    save_json(directory / "test_metrics.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
