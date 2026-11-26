"""Serve the local workspace and its inference endpoints on loopback."""

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.util import find_spec
import json
import logging
import os
from pathlib import Path
from secrets import randbelow
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".model-cache"
os.environ.setdefault("HF_HOME", str(CACHE / "huggingface"))

from models import DOODLE_GENERATORS, LLM, ModelError, generate_image, generate_prompt, rewrite_post, suggest, training_status
from references import find_references
from image_api import generate_api_image, image_status, load_environment
from scratch.config import CATEGORIES

load_environment()

DEPENDENCIES = ("accelerate", "numpy", "PIL", "torch", "transformers")
FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


def missing_dependencies():
    return [package for package in DEPENDENCIES if find_spec(package) is None]


def field(data, name, limit):
    value = data.get(name)
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= limit:
        raise ValueError(f"{name} 必须是 1–{limit} 个字符。")
    return value.strip()


class WorkspaceHandler(BaseHTTPRequestHandler):
    def send_bytes(self, content, media_type, status=HTTPStatus.OK, **headers):
        self.send_response(status)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob: https://i.imgflip.com; frame-ancestors 'none'")
        self.send_header("Referrer-Policy", "no-referrer")
        for name, value in headers.items():
            self.send_header(name, str(value))
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, data, status=HTTPStatus.OK):
        self.send_bytes(json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

    def allowed_request(self):
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        host = self.headers.get("Host")
        origin = self.headers.get("Origin")
        if host in hosts and (origin is None or origin == f"http://{host}"):
            return True
        self.send_json({"error": "请从本地工作台发起请求。"}, HTTPStatus.FORBIDDEN)
        return False

    def do_GET(self):
        if not self.allowed_request():
            return
        path = urlsplit(self.path).path
        if path in FILES:
            filename, media_type = FILES[path]
            self.send_bytes((ROOT / "dist" / filename).read_bytes(), media_type)
        elif path == "/api/health":
            self.send_json({
                "missing": missing_dependencies(),
                "models": {"encoder": "scratch EmotionEncoder", "llm": LLM,
                           "generator": "scratch ConditionalVAE", "classifier": "scratch DoodleCNN",
                           "prompt": "scratch GRU attention/copy encoder-decoder",
                           "scene": "external OpenAI GPT Image API"},
                "checkpoints": training_status(),
                "image_api": image_status(),
            })
        else:
            self.send_json({"error": "路径不存在。"}, HTTPStatus.NOT_FOUND)

    def read_payload(self):
        if self.headers.get_content_type() != "application/json":
            raise ValueError("请求必须使用 JSON。")
        size = int(self.headers.get("Content-Length", "0"))
        if not 1 <= size <= 16384:
            raise ValueError("请求长度超出范围。")
        try:
            data = json.loads(self.rfile.read(size))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("请求不是有效的 JSON。") from error
        if not isinstance(data, dict):
            raise ValueError("请求必须是 JSON 对象。")
        return data

    def do_POST(self):
        if not self.allowed_request():
            return
        path = urlsplit(self.path).path
        if path not in ("/api/suggest", "/api/image", "/api/prompt", "/api/gpt-image", "/api/rewrite", "/api/references"):
            self.send_json({"error": "路径不存在。"}, HTTPStatus.NOT_FOUND)
            return

        try:
            data = self.read_payload()
            if path == "/api/references":
                self.send_json(find_references(field(data, "post", 1000)))
                return
            if path == "/api/rewrite":
                post = field(data, "post", 1000)
                style = data.get("style", "expand")
                if not isinstance(style, str) or style not in ("expand", "humor", "concise"):
                    raise ValueError("请选择有效的改写风格。")
                self.ensure_dependencies()
                self.send_json(rewrite_post(post, style))
                return
            if path == "/api/prompt":
                post = field(data, "post", 1000)
                self.ensure_dependencies()
                self.send_json(generate_prompt(post))
                return
            if path == "/api/gpt-image":
                prompt = field(data, "prompt", 4000)
                image, model = generate_api_image(prompt)
                self.send_bytes(image, "image/png", **{"X-Image-Model": model})
                return
            if path == "/api/suggest":
                post = field(data, "post", 1000)
                use_sentiment = data.get("use_sentiment", True)
                if type(use_sentiment) is not bool:
                    raise ValueError("情绪参考选项必须是布尔值。")
                self.ensure_dependencies()
                result = suggest(post, use_sentiment)
                self.send_json(result)
                return

            category = data.get("category")
            color = data.get("color")
            mode = data.get("mode")
            seed = data.get("seed")
            generator = data.get("generator", "vae")
            if category not in CATEGORIES or color not in ("#5936b8", "#25203c", "#d94f88") or mode not in ("cover", "sticker"):
                raise ValueError("请选择有效的涂鸦类别、颜色和图片类型。")
            if generator not in DOODLE_GENERATORS:
                raise ValueError("请选择有效的涂鸦生成模型。")
            if seed is None:
                seed = randbelow(2**32)
            if type(seed) is not int or not 0 <= seed < 2**32:
                raise ValueError("种子必须是 0 至 4294967295 之间的整数。")
            self.ensure_dependencies()
            image, score = generate_image(category, color, mode == "sticker", seed, generator=generator)
            self.send_bytes(image, "image/png", **{"X-Image-Seed": seed, "X-Category-Score": f"{score:.4f}"})
        except (BrokenPipeError, ConnectionResetError):
            return
        except ValueError as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except ModelError as error:
            logging.error("Inference failed (%s): %s", path, error.__cause__ or type(error).__name__)
            self.send_json({"error": str(error)}, HTTPStatus.SERVICE_UNAVAILABLE)
        except Exception:
            logging.exception("Unexpected inference error (%s)", path)
            self.send_json({"error": "本地推理失败，请查看启动终端。"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    @staticmethod
    def ensure_dependencies():
        missing = missing_dependencies()
        if missing:
            raise ModelError("缺少本地依赖：" + ", ".join(missing) + "。请按运行说明安装。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), WorkspaceHandler) as server:
        print(f"PostPalette: http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
