"""Offline HTTP tests for /api/image, /api/suggest and /api/prompt.

Model functions are patched where server.py imported them; missing-weight cases
use an empty temporary ARTIFACTS directory and the real checkpoint checks.
"""

from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import models
from models import ModelError
from server import WorkspaceHandler

PNG = b"\x89PNG\r\n\x1a\nfixture"
JSON = {"Content-Type": "application/json"}
DOODLE = {"category": "cat", "color": "#25203c", "mode": "sticker", "seed": 2026}


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), WorkspaceHandler)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()

        def stop():
            self.server.shutdown()
            self.server.server_close()
            thread.join()

        self.addCleanup(stop)
        # Route tests must not depend on which optional packages the machine has.
        patcher = patch("server.missing_dependencies", return_value=[])
        patcher.start()
        self.addCleanup(patcher.stop)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        patcher = patch.object(models, "ARTIFACTS", Path(directory.name))
        patcher.start()
        self.addCleanup(patcher.stop)

    def post(self, path, body):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        self.addCleanup(connection.close)
        connection.request("POST", path, json.dumps(body), JSON)
        response = connection.getresponse()
        return response, response.read()

    def test_image_returns_png_with_seed_and_score(self):
        with patch("server.generate_image", return_value=(PNG, 0.9123)) as generate:
            response, content = self.post("/api/image", DOODLE)
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("Content-Type"), "image/png")
        self.assertEqual(content, PNG)
        self.assertEqual(response.getheader("X-Image-Seed"), "2026")
        self.assertEqual(response.getheader("X-Category-Score"), "0.9123")
        generate.assert_called_once_with("cat", "#25203c", True, 2026)

    def test_image_without_seed_picks_one(self):
        with patch("server.generate_image", return_value=(PNG, 0.5)) as generate:
            response, _ = self.post("/api/image", {**DOODLE, "seed": None})
        self.assertEqual(response.status, 200)
        seed = int(response.getheader("X-Image-Seed"))
        self.assertEqual(generate.call_args.args[3], seed)
        self.assertTrue(0 <= seed < 2**32)

    def test_image_rejects_invalid_input(self):
        cases = [{**DOODLE, "category": "dog"}, {**DOODLE, "color": "#000000"},
                 {**DOODLE, "mode": "poster"}, {**DOODLE, "seed": -1}, {**DOODLE, "seed": 2**32},
                 {**DOODLE, "seed": "7"}, {**DOODLE, "seed": True}]
        with patch("server.generate_image") as generate:
            for body in cases:
                with self.subTest(body=body):
                    response, content = self.post("/api/image", body)
                    self.assertEqual(response.status, 400)
                    self.assertIn("error", json.loads(content))
        generate.assert_not_called()

    def test_image_without_weights_is_503_with_reason(self):
        response, content = self.post("/api/image", DOODLE)
        self.assertEqual(response.status, 503)
        self.assertIn("vae", json.loads(content)["error"])

    def test_suggest_without_emotion_weights_returns_degraded_result(self):
        with patch.object(models, "language_model", return_value=(None, None)), \
                patch.object(models, "shortlist", return_value=[{"name": f"t{n}", "use": "u"} for n in range(5)]), \
                patch.object(models, "complete", return_value='{"meme_numbers":[1,2,3],"category":"sun","caption":"Hi"}'):
            response, content = self.post("/api/suggest", {"post": "sunny day", "use_sentiment": True})
        self.assertEqual(response.status, 200)
        result = json.loads(content)
        self.assertTrue(result["degraded"])
        self.assertFalse(result["emotion_used"])
        self.assertEqual(result["category"], "sun")
        self.assertEqual(len(result["memes"]), 3)

    def test_suggest_rejects_non_boolean_sentiment_flag(self):
        with patch("server.suggest") as suggest:
            response, _ = self.post("/api/suggest", {"post": "hi", "use_sentiment": "yes"})
        self.assertEqual(response.status, 400)
        suggest.assert_not_called()

    def test_prompt_model_error_is_503_not_500(self):
        with patch("server.generate_prompt", side_effect=ModelError("提示词生成失败")):
            response, content = self.post("/api/prompt", {"post": "hello"})
        self.assertEqual(response.status, 503)
        self.assertEqual(json.loads(content)["error"], "提示词生成失败")

    def test_prompt_without_weights_is_503(self):
        response, content = self.post("/api/prompt", {"post": "hello"})
        self.assertEqual(response.status, 503)
        self.assertIn("prompt", json.loads(content)["error"])


if __name__ == "__main__":
    unittest.main()
