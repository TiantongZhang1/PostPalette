"""CPU-only/offline regression tests for API wiring and prompt provenance."""

import base64
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from io import BytesIO
import json
import os
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import image_api
from models import ModelError
from server import WorkspaceHandler


PNG = b"\x89PNG\r\n\x1a\nfixture"


class Response:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, limit):
        return json.dumps({"data": [{"b64_json": base64.b64encode(PNG).decode()}]}).encode()


class ImagePipelineTests(unittest.TestCase):
    def test_credentials_remain_on_server(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-secret", "POSTPALETTE_IMAGE_MODEL": "gpt-image-1.5", "POSTPALETTE_IMAGE_QUALITY": "medium"}):
            with patch.object(image_api, "urlopen", return_value=Response()) as send:
                content, model = image_api.generate_api_image("friends having a barbecue")
            self.assertEqual(content, PNG)
            request = send.call_args.args[0]
            payload = json.loads(request.data)
            self.assertEqual(payload["prompt"], "friends having a barbecue")
            self.assertEqual(payload["output_format"], "png")
            self.assertEqual(payload["n"], 1)
            self.assertNotIn("test-secret", json.dumps(image_api.image_status()))

    def test_missing_key_never_sends_request(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}), patch.object(image_api, "urlopen") as send:
            with self.assertRaisesRegex(ModelError, "OPENAI_API_KEY"):
                image_api.generate_api_image("a scene")
            send.assert_not_called()

    def test_api_error_does_not_echo_provider_details(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-secret"}), patch.object(image_api, "urlopen", side_effect=HTTPError("https://api.openai.com", 401, "test-secret", {}, None)):
            with self.assertRaises(ModelError) as caught:
                image_api.generate_api_image("a scene")
            self.assertNotIn("test-secret", str(caught.exception))

    def test_provider_errors_are_logged_without_credentials(self):
        key = "sk-proj-REALSECRET1234567890abcd"
        body = json.dumps({"error": {
            "message": f"Incorrect API key provided: sk-proj-****abcd. Sent {key}",
            "type": "invalid_request_error", "code": "invalid_api_key"}}).encode()
        failures = [
            HTTPError("https://api.openai.com", 401, "Unauthorized", {"x-request-id": "req_42"}, BytesIO(body)),
            HTTPError("https://api.openai.com", 502, "Bad Gateway", {}, BytesIO(b"<html>Bearer " + key.encode() + b"</html>")),
            URLError(TimeoutError("timed out")),
        ]
        server = ThreadingHTTPServer(("127.0.0.1", 0), WorkspaceHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.dict(os.environ, {"OPENAI_API_KEY": key}), self.assertLogs(level="ERROR") as logs:
                for failure in failures:
                    with patch.object(image_api, "urlopen", side_effect=failure):
                        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
                        connection.request("POST", "/api/gpt-image", json.dumps({"prompt": "a scene"}),
                                           {"Content-Type": "application/json"})
                        response = connection.getresponse()
                        content = response.read().decode()
                        connection.close()
                    self.assertEqual(response.status, 503)
                    self.assertNotIn("REALSECRET", content)
                    self.assertNotIn("invalid_api_key", content, "provider details stay in the server log")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        log = "\n".join(logs.output)
        self.assertIn("OpenAI HTTP 401", log)
        self.assertIn("req_42", log)
        self.assertIn("invalid_api_key", log)
        self.assertIn("OpenAI HTTP 502", log)
        self.assertIn("URLError", log)
        for secret in ("REALSECRET", "****abcd", key):
            self.assertNotIn(secret, log)

    def test_error_keeps_original_cause(self):
        error = HTTPError("https://api.openai.com", 403, "Forbidden", {}, BytesIO(b"{}"))
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-secret"}), patch.object(image_api, "urlopen", side_effect=error):
            with self.assertRaises(ModelError) as caught:
                image_api.generate_api_image("a scene")
        self.assertIs(caught.exception.__cause__, error)
        self.assertIn("OpenAI HTTP 403", caught.exception.log_detail)

    def test_http_routes_and_origin_validation(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), WorkspaceHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        try:
            connection.request("GET", "/api/health")
            response = connection.getresponse()
            health = json.loads(response.read())
            self.assertIn("prompt", health["checkpoints"])
            self.assertIn("configured", health["image_api"])
            connection.request("POST", "/api/scene", "{}", {"Content-Type": "application/json"})
            response = connection.getresponse()
            self.assertEqual(response.status, 404)
            response.read()
            with patch("server.generate_prompt", return_value={"draft": "friends at a grill", "prompt": "reviewable prompt", "step": 1200}) as writer:
                connection.request("POST", "/api/prompt", json.dumps({"post": "friends having a barbecue"}), {"Content-Type": "application/json"})
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(json.loads(response.read())["draft"], "friends at a grill")
                writer.assert_called_once_with("friends having a barbecue")
            with patch("server.generate_api_image", return_value=(PNG, "gpt-image-1.5")) as generate:
                connection.request("POST", "/api/gpt-image", json.dumps({"prompt": "a barbecue"}), {"Content-Type": "application/json"})
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(response.read(), PNG)
                generate.assert_called_once_with("a barbecue")
                connection.request("POST", "/api/gpt-image", json.dumps({"prompt": "a barbecue"}), {"Content-Type": "application/json", "Origin": "https://example.com"})
                response = connection.getresponse()
                self.assertEqual(response.status, 403)
                response.read()
                self.assertEqual(generate.call_count, 1)
            connection.request("POST", "/api/gpt-image", json.dumps({"prompt": " "}), {"Content-Type": "application/json"})
            response = connection.getresponse()
            self.assertEqual(response.status, 400)
            response.read()
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
