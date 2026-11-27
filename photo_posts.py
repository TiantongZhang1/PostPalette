"""Uploaded photos -> scratch CNN tags -> user confirmation -> text-only Qwen."""

from functools import cache

from models import ModelError, inference_lock, rewrite_post
from scratch.config import ARTIFACTS
from scratch.photo_tags import LABELS, NAMES, PhotoTagCNN, decode_photo
from scratch.daily_activity import NAMES as ACTIVITY_NAMES


def photo_status():
    return {"ready": (ARTIFACTS / "photo_tags/best.pt").is_file(),
            "labels": [{"id": key, "name": NAMES[key]} for key in LABELS],
            "device": "cpu", "native_size": 64}


@cache
def photo_model():
    import torch
    path = ARTIFACTS / "photo_tags/best.pt"
    if not path.is_file():
        raise ModelError("照片 CNN 尚未训练。请按 README 准备并训练 photo_tags；不会用预训练视觉模型替代。")
    record = torch.load(path, map_location="cpu", weights_only=True)
    if (record.get("task") != "photo_tags" or record.get("initialization") != "random"
            or record.get("labels") != list(LABELS) or record.get("step", 0) < 1):
        raise ModelError("照片模型的训练记录或 80 类标签不兼容。")
    model = PhotoTagCNN(**record["config"]).cpu().eval()
    model.load_state_dict(record["model"], strict=True)
    return model, record


def recognize_photo(content):
    pixels = decode_photo(content)
    import torch
    with inference_lock, torch.inference_mode():
        torch.set_num_threads(4)
        model, record = photo_model()
        probabilities = model(torch.from_numpy(pixels)[None].float() / 255).sigmoid()[0].tolist()
        counts = record["manifest"]["splits"]
        predictions = []
        for i, key in enumerate(LABELS):
            low_support = counts["train"]["positive_counts"][key] < 100 or counts["val"]["positive_counts"][key] < 5
            threshold = record.get('suggestion_thresholds', [max(0.6, value) for value in record['thresholds']])[i]
            predictions.append({"id": key, "name": NAMES[key], "score": probabilities[i],
                                "suggested": probabilities[i] >= threshold and not low_support,
                                "auto_supported": threshold <= 1 and not low_support,
                                "low_support": low_support})
        predictions.sort(key=lambda item: -item["score"])
        # Avoid filling the writing prompt with a long uncertain list.
        selected = 0
        for item in predictions:
            if item["suggested"]:
                selected += 1
                item["suggested"] = selected <= 5
        return {"predictions": predictions, "step": record["step"], "device": "cpu", "native_size": 64,
                "notice": "实验 CNN 只判断 80 类物体是否出现，不识别人物关系、地点、数量或活动。分数未经校准，请逐项确认。"}


def write_photo_post(labels, context, style, confirmed, activity=None, activity_confirmed=False):
    if type(activity_confirmed) is not bool:
        raise ValueError("活动确认状态必须为布尔值。")
    if activity is not None and (not isinstance(activity,str) or activity not in ACTIVITY_NAMES):
        raise ValueError("请选择支持的日常活动。")
    if activity is not None and activity_confirmed is not True:
        raise ValueError("请先核对并确认日常活动。")
    if activity is None and activity_confirmed:
        raise ValueError("请选择活动后再确认。")
    if (not isinstance(labels, list) or not 0 <= len(labels) <= 20
            or any(not isinstance(key, str) or key not in NAMES for key in labels)):
        raise ValueError("请确认最多 20 个照片物体标签。")
    if not isinstance(context, str) or len(context) > 500:
        raise ValueError("补充信息不能超过 500 个字符。")
    if not labels and activity is None and not context.strip():
        raise ValueError("请确认至少一个物体、日常活动或补充说明。")
    if confirmed is not True:
        raise ValueError("请先核对并确认照片线索。")
    if not isinstance(style, str) or style not in ("expand", "humor", "concise"):
        raise ValueError("请选择有效的推文风格。")
    labels = list(dict.fromkeys(labels))
    facts = "照片中可见：" + "、".join(NAMES[key] for key in labels) + "。" if labels else ""
    if activity is not None:
        facts += "用户确认照片中的主要活动：" + ACTIVITY_NAMES[activity] + "。人数与人物关系未由此确定。"
    if context.strip():
        facts += "用户补充：" + context.strip()
    result = rewrite_post(facts, style, photo=True)
    result["confirmed_labels"] = labels
    result["confirmed_activity"] = activity
    result["notice"] = "Qwen 仅根据你确认的活动、标签和补充写作，没有看到照片。请检查是否添加了未经提供的事实。"
    return result
