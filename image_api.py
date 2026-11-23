"""Server-side OpenAI image adapter. Never expose credentials to the browser."""

import base64
import binascii
import json
import os
from pathlib import Path
from threading import Lock
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from models import ModelError

_lock = Lock()


def load_environment():
    path = Path(__file__).resolve().parent / ".env"
    if not path.is_file():
        return
    allowed = {"OPENAI_API_KEY", "POSTPALETTE_IMAGE_MODEL", "POSTPALETTE_IMAGE_QUALITY"}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        name, separator, value = line.strip().partition("=")
        if separator and name.strip() in allowed:
            os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


def image_status():
    return {"configured": bool(os.environ.get("OPENAI_API_KEY", "").strip()),
            "model": os.environ.get("POSTPALETTE_IMAGE_MODEL", "gpt-image-1.5"),
            "quality": os.environ.get("POSTPALETTE_IMAGE_QUALITY", "medium")}


def generate_api_image(prompt):
    config = image_status()
    if not config["configured"]:
        raise ModelError("尚未配置 OPENAI_API_KEY。请在项目 .env 中填写后重启服务；不要把密钥填到网页或聊天中。")
    if not config["model"].startswith("gpt-image-") or config["quality"] not in ("low", "medium", "high", "auto"):
        raise ModelError("请检查 .env 中的 GPT Image 模型名与质量配置。")
    payload = {"model": config["model"], "prompt": prompt, "n": 1,
               "size": "1024x1024", "quality": config["quality"], "output_format": "png"}
    request = Request("https://api.openai.com/v1/images/generations",
                      data=json.dumps(payload).encode("utf-8"), method="POST", headers={
                          "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"].strip(),
                          "Content-Type": "application/json"})
    if not _lock.acquire(blocking=False):
        raise ModelError("已有一张图片正在生成，请等待完成后重试。")
    try:
        with urlopen(request, timeout=180) as response:
            content = response.read(32 * 1024 * 1024 + 1)
        if len(content) > 32 * 1024 * 1024:
            raise ValueError("Image response exceeds limit")
        result = json.loads(content)
        image = base64.b64decode(result["data"][0]["b64_json"], validate=True)
        if not image.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Expected PNG")
        return image, config["model"]
    except HTTPError as error:
        messages = {400: "图像请求被拒绝，请调整描述或检查模型配置。", 401: "API 密钥无效，请检查服务端 .env。",
                    403: "此 API 项目暂无图像模型访问权限，请检查账号权限或组织验证。",
                    404: "图像模型不可用，请检查配置的模型名。", 429: "API 额度不足或请求过快，请检查额度并稍后重试。"}
        raise ModelError(messages.get(error.code, "OpenAI 图像服务暂时不可用，请稍后重试。")) from None
    except (TimeoutError, URLError):
        raise ModelError("连接 OpenAI 超时或中断。请求可能已处理，请检查 API 用量后再决定是否重试。") from None
    except (ValueError, KeyError, IndexError, TypeError, binascii.Error):
        raise ModelError("图像服务返回了无法读取的图片。") from None
    finally:
        _lock.release()
