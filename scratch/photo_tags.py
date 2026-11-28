"""Scratch photo tagging and shared, nonlearned image preprocessing."""

from io import BytesIO
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from torch import nn

NAMES = {
    "person": "人物", "bicycle": "自行车", "car": "汽车", "motorcycle": "摩托车", "airplane": "飞机",
    "bus": "公交车", "train": "火车", "truck": "卡车", "boat": "船", "traffic light": "交通灯",
    "fire hydrant": "消防栓", "stop sign": "停车标志", "parking meter": "停车计时器", "bench": "长椅",
    "bird": "鸟", "cat": "猫", "dog": "狗", "horse": "马", "sheep": "羊", "cow": "牛",
    "elephant": "大象", "bear": "熊", "zebra": "斑马", "giraffe": "长颈鹿", "backpack": "背包",
    "umbrella": "雨伞", "handbag": "手提包", "tie": "领带", "suitcase": "行李箱", "frisbee": "飞盘",
    "skis": "滑雪板（双板）", "snowboard": "滑雪板（单板）", "sports ball": "运动球", "kite": "风筝",
    "baseball bat": "棒球棒", "baseball glove": "棒球手套", "skateboard": "滑板", "surfboard": "冲浪板",
    "tennis racket": "网球拍", "bottle": "瓶子", "wine glass": "高脚杯", "cup": "杯子", "fork": "叉子",
    "knife": "刀", "spoon": "勺子", "bowl": "碗", "banana": "香蕉", "apple": "苹果", "sandwich": "三明治",
    "orange": "橙子", "broccoli": "西兰花", "carrot": "胡萝卜", "hot dog": "热狗", "pizza": "披萨",
    "donut": "甜甜圈", "cake": "蛋糕", "chair": "椅子", "couch": "沙发", "potted plant": "盆栽",
    "bed": "床", "dining table": "餐桌", "toilet": "马桶", "tv": "电视", "laptop": "笔记本电脑",
    "mouse": "鼠标", "remote": "遥控器", "keyboard": "键盘", "cell phone": "手机", "microwave": "微波炉",
    "oven": "烤箱", "toaster": "烤面包机", "sink": "水槽", "refrigerator": "冰箱", "book": "书",
    "clock": "时钟", "vase": "花瓶", "scissors": "剪刀", "teddy bear": "玩具熊", "hair drier": "吹风机",
    "toothbrush": "牙刷",
}
GROUPS = {key: (name, [key]) for key, name in NAMES.items()}
LABELS = tuple(GROUPS)
MAX_UPLOAD = 8 * 1024 * 1024


def decode_photo(content):
    if not content or len(content) > MAX_UPLOAD:
        raise ValueError("请选择不超过 8 MB 的照片。")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as source:
                if source.format not in ("JPEG", "PNG", "WEBP"):
                    raise ValueError("仅支持 JPEG、PNG 和 WebP 照片。")
                if source.width * source.height > 20_000_000 or min(source.size) < 16:
                    raise ValueError("照片需至少 16×16 像素，且不超过 2000 万像素。")
                if getattr(source, "n_frames", 1) != 1:
                    raise ValueError("请使用静态照片，不支持动画。")
                image = ImageOps.exif_transpose(source).convert("RGBA")
                background = Image.new("RGBA", image.size, (127, 127, 127, 255))
                image = Image.alpha_composite(background, image).convert("RGB")
                image = ImageOps.pad(image, (64, 64), method=Image.Resampling.LANCZOS, color=(127, 127, 127))
                return np.asarray(image).transpose(2, 0, 1).copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ValueError("照片无法读取或尺寸过大，请换一张静态 JPEG/PNG/WebP。") from error


class PhotoTagCNN(nn.Module):
    """All convolutions, normalization and output weights are randomly initialized."""
    def __init__(self, classes=len(LABELS), width=16):
        super().__init__()
        blocks = []
        previous = 3
        for channels in (width, width * 2, width * 4, width * 6):
            blocks.extend([nn.Conv2d(previous, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(), nn.MaxPool2d(2)])
            previous = channels
        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.15), nn.Linear(previous, classes))

    def forward(self, images):
        return self.head(self.features(images))
