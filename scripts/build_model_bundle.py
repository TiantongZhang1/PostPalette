"""Build allowlisted Release assets; validate every checkpoint on CPU only.

Run after committing the matching source:
    python scripts/build_model_bundle.py --include-experiments
Only best.pt is distributed. last.pt and original datasets stay local.
"""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scratch.config import CATEGORIES, EMOTIONS
from scratch.daily_activity import ActivityCNN
from scratch.networks import create_network
from scratch.people_networks import ImageClassifier, PersonPoseNet
from scratch.experiments.photo_diffusion import PhotoUNet
from scratch.photo_tags import PhotoTagCNN
from scratch.prompt_model import PromptWriter

CURRENT = ("emotion", "cnn", "vae", "gan", "prompt", "photo_tags", "person_pose", "expression", "daily_activity")
EXPERIMENTS = ("action", "scene")


def validate(task, payload):
    record = torch.load(BytesIO(payload), map_location="cpu", weights_only=True)
    assert record["task"] == task and record["initialization"] == "random", task
    assert record.get("step", record.get("trained_updates", 0)) > 0, task
    if task in ("emotion", "cnn", "vae", "gan"):
        assert record["labels"] == list(EMOTIONS if task == "emotion" else CATEGORIES)
        model = create_network(task, record["config"])
    elif task == "prompt":
        model = PromptWriter(**record["config"])
    elif task == "photo_tags":
        model = PhotoTagCNN(**record["config"])
    elif task == "person_pose":
        model = PersonPoseNet()
    elif task in ("action", "expression"):
        model = ImageClassifier(**record["config"])
    elif task == "daily_activity":
        model = ActivityCNN(**record["config"])
    else:
        model = PhotoUNet(**record["config"])
    model.cpu().eval().load_state_dict(record["ema"] if task == "scene" else record["model"], strict=True)
    with torch.inference_mode():
        if task == "emotion":
            output = model(torch.ones((1, 96), dtype=torch.long))
        elif task == "cnn":
            output = model(torch.zeros(1, 1, 28, 28))
        elif task in ("vae", "gan"):
            noise, label = torch.zeros(1, model.latent), torch.zeros(1, dtype=torch.long)
            output = model.generate(noise, label) if task == "gan" else model.decode(noise, label)
        elif task == "prompt":
            output = model(torch.tensor([[4, 3, 0, 0]]), torch.tensor([[2, 4, 3]]))
        elif task == "scene":
            config = record["config"]
            tokens = (torch.zeros(1, config["length"], config["feature_dim"]) if config.get("feature_dim")
                      else torch.ones(1, config["length"], dtype=torch.long))
            output = model(torch.zeros(1, 3, record["size"], record["size"]), torch.ones(1), tokens)
        else:
            size = {"photo_tags": 64, "expression": 48, "action": 128}.get(task, record.get("size", 160))
            output = model(torch.zeros(1, 1 if task == "expression" else 3, size, size))
        for value in output if isinstance(output, tuple) else (output,):
            assert torch.isfinite(value).all(), task
    metadata = {key: record.get(key) for key in ("task", "initialization", "config", "size", "labels", "epoch", "step", "trained_updates", "data_hash")}
    metadata["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    return metadata, record.get("history", [])


def build(tasks, filename, commit, tag, experimental=False):
    files, models = {}, {}
    for task in tasks:
        directory = ROOT / "artifacts" / task
        payload = (directory / "best.pt").read_bytes()
        metadata, checkpoint_history = validate(task, payload)
        files[f"artifacts/{task}/best.pt"] = payload
        models[task] = metadata
        # Small public training records only; never logs, user images or configs containing secrets.
        for name in ("history.json", "test_metrics.json", "calibration.json", "selection.json", "validation_summary.json"):
            path = directory / name
            if not path.exists():
                continue
            content = path.read_bytes()
            try:
                json.loads(content)
            except (ValueError, UnicodeDecodeError):
                if name != "history.json":
                    raise
                content = json.dumps(checkpoint_history, ensure_ascii=False, indent=2).encode("utf-8")
                metadata["history_source"] = "checkpoint history; invalid standalone history excluded"
            files[f"artifacts/{task}/{name}"] = content
        if task in ("vae", "gan"):
            files[f"artifacts/{task}/samples.png"] = (directory / "samples.png").read_bytes()
        print(f"Validated on CPU: {task}", flush=True)
    manifest = {"tag": tag, "source_commit": commit, "source_repository": "https://github.com/TiantongZhang1/PostPalette",
                "experimental": experimental, "models": models,
                "validation": "weights_only CPU load, strict state_dict, finite forward output, per-file SHA-256 and ZIP CRC",
                "files": [{"path": name, "bytes": len(data), "sha256": sha256(data).hexdigest()} for name, data in files.items()]}
    notes = f"""PostPalette {tag} — {'历史实验包（可选）' if experimental else '当前功能模型包'}
对应源码：{commit}
模型：{', '.join(tasks)}

在包含 server.py 的源码根目录解压。已有同名模型时先备份。
PowerShell: Expand-Archive -LiteralPath .\\{filename} -DestinationPath . -Force
正确路径示例：artifacts/{tasks[0]}/best.pt，不要额外套一层 artifacts。
按 README.txt 安装依赖，运行 .\\.venv\\Scripts\\python.exe server.py。
服务已运行时需重启以清除旧模型缓存。

包内为原始 best.pt（含原检查点训练元数据）、训练/评估记录及可用的 VAE/GAN 采样图。
MODEL_MANIFEST.json 记录每个模型与文件的校验值；ZIP 同页的 .sha256 用于整体校验。
只发布 best.pt，没有 last.pt 或原始数据集；从最新批次续训仍需另行共享 last.pt 与相同数据划分。
不包含 .env、API 密钥、Qwen 权重、本机缓存或用户上传图片。Qwen 首次使用需联网下载。
GPT 场景图片需自行配置 OPENAI_API_KEY；其他功能无需该密钥。外部生图的课程许可需单独确认。
模型输出需人工核对，指标不代表生成图片美观程度或真实场景中的可靠性。
"""
    if experimental:
        notes += "\n旧 action（Stanford40）已退出默认流程；scene 为暂停的早期扩散实验，不能保证生成质量。\n普通使用只需当前功能模型包；不要用本实验包替换 daily_activity。\n"
    files["MODEL_INSTALL.txt"] = notes.encode("utf-8")
    manifest["files"].append({"path": "MODEL_INSTALL.txt", "bytes": len(files["MODEL_INSTALL.txt"]), "sha256": sha256(files["MODEL_INSTALL.txt"]).hexdigest()})
    files["MODEL_MANIFEST.json"] = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    output = ROOT / "artifacts" / "releases"
    output.mkdir(parents=True, exist_ok=True)
    archive = output / filename
    if archive.exists():
        raise FileExistsError(f"Refusing to replace an existing bundle: {archive}")
    with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=6) as bundle:
        for name, data in files.items():
            bundle.writestr(name, data)
    with ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert set(bundle.namelist()) == set(files)
        for entry in manifest["files"]:
            assert sha256(bundle.read(entry["path"])).hexdigest() == entry["sha256"]
    digest = sha256(archive.read_bytes()).hexdigest()
    (output / (filename + ".sha256")).write_text(digest + "  " + filename + "\n", encoding="ascii")
    (output / (filename + ".manifest.json")).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"archive": filename, "bytes": archive.stat().st_size, "sha256": digest, "models": list(models), "source_commit": commit}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="models-v2")
    parser.add_argument("--include-experiments", action="store_true")
    args = parser.parse_args()
    if not args.tag.replace("-", "").isalnum():
        parser.error("tag must contain only letters, numbers and hyphens")
    torch.set_num_threads(4)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    build(CURRENT, f"postpalette-{args.tag}.zip", commit, args.tag)
    if args.include_experiments:
        build(EXPERIMENTS, f"postpalette-experiments-{args.tag.removeprefix('models-')}.zip", commit, args.tag, True)


if __name__ == "__main__":
    main()
