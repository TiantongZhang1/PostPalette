"""Full-directory retrieval, offline continuity, exclusion and HTTP contracts."""

from contextlib import contextmanager
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import memes
import models
from server import WorkspaceHandler


@contextmanager
def isolated_catalog():
    with tempfile.TemporaryDirectory() as directory:
        with patch.object(memes, "CACHE_FILE", Path(directory) / "cache.json"), \
             patch.object(memes, "_catalog_state", None), patch.object(memes, "_catalog_until", 0):
            yield


class MemeTests(unittest.TestCase):
    def test_live_new_template_participates_and_is_saved(self):
        new = {"id": "987654321", "name": "New Barbecue Template", "url": "https://i.imgflip.com/abc123.jpg"}
        response = BytesIO(json.dumps({"success": True, "data": {"memes": [new]}}).encode())
        with isolated_catalog(), patch.object(memes, "urlopen", return_value=response):
            catalog = memes.load_catalog()
            self.assertEqual(catalog["mode"], "live")
            self.assertGreaterEqual(len(catalog["templates"]), 101)
            self.assertIn(new["id"], [row["id"] for row in memes.shortlist("barbecue", catalog=catalog)])
            self.assertIn(new["id"], [row["id"] for row in json.loads(memes.CACHE_FILE.read_text())["memes"]])

    def test_offline_bundled_catalog_does_not_shrink_to_22(self):
        with isolated_catalog(), patch.object(memes, "urlopen", side_effect=OSError("offline")) as fetch:
            catalog = memes.load_catalog()
            self.assertEqual(catalog["mode"], "bundled")
            self.assertEqual(len(catalog["templates"]), 100)
            self.assertIs(memes.load_catalog(), catalog)
            self.assertEqual(fetch.call_count, 1)

    def test_cache_keeps_a_past_template_when_network_fails(self):
        with isolated_catalog(), patch.object(memes, "urlopen", side_effect=OSError("offline")):
            memes.CACHE_FILE.write_text(json.dumps({"memes": [{"id": "987654321", "name": "Past popular", "url": "https://i.imgflip.com/abc123.png"}]}))
            catalog = memes.load_catalog()
            self.assertEqual(catalog["mode"], "cached")
            self.assertEqual(len(catalog["templates"]), 101)

    def test_invalid_remote_urls_and_duplicate_ids_are_rejected(self):
        rows = [
            {"id": "1", "name": "good", "url": "https://i.imgflip.com/abc.jpg"},
            {"id": "1", "name": "duplicate", "url": "https://i.imgflip.com/def.jpg"},
            {"id": "2", "name": "bad", "url": "https://evil.example/abc.jpg"},
            {"id": "3", "name": "bad", "url": "javascript:alert(1)"},
            {"id": "../4", "name": "bad", "url": "https://i.imgflip.com/abc.jpg"},
            None,
        ]
        self.assertEqual(len(memes._valid_rows({"memes": rows})), 1)
        self.assertEqual(memes._valid_rows({"memes": {}}), [])

    def test_entire_catalog_is_annotated_and_refresh_excludes_seen_ids(self):
        with isolated_catalog(), patch.object(memes, "urlopen", side_effect=OSError("offline")):
            catalog = memes.load_catalog()
        self.assertTrue(all(row["terms"] for row in catalog["templates"]))
        first = memes.shortlist("barbecue food with friends", catalog=catalog)
        self.assertEqual(len(first), 24)
        seen = [row["id"] for row in first]
        second = memes.shortlist("barbecue food with friends", seen, catalog=catalog)
        self.assertFalse(set(seen) & {row["id"] for row in second})
        # IDs outside the original 22 must actually reach the language model.
        self.assertTrue({row["id"] for row in first} - {row[0] for row in memes.TEMPLATES})
        for post, expected in [("thirsty and hungry for another meal", "124055727"),
                               ("we drank a toast to celebrate graduation", "5496396")]:
            with patch.object(memes, "randbits", return_value=123):
                self.assertIn(expected, {row["id"] for row in memes.shortlist(post, catalog=catalog)})

    def test_exclusion_validation_rejects_bad_input(self):
        self.assertEqual(memes.validate_exclusions(["1", "1", "2"]), ["1", "2"])
        for bad in (None, "1", [1], [True], ["https://example.com"], ["1"] * 121):
            with self.assertRaises(ValueError):
                memes.validate_exclusions(bad)

    def test_suggest_forwards_full_candidates_and_never_returns_excluded_templates(self):
        with isolated_catalog(), patch.object(memes, "urlopen", side_effect=OSError("offline")):
            catalog = memes.load_catalog()
        output = '{"meme_numbers":[1,2,3],"category":"cat","caption":"Dinner time!"}'
        with patch.object(models, "load_catalog", return_value=catalog), \
             patch.object(models, "language_model", return_value=(None, None)), \
             patch.object(models, "complete", return_value=output) as complete:
            first = models.suggest("barbecue with friends", False)
            excluded = [row["id"] for row in first["memes"]]
            second = models.suggest("barbecue with friends", False, excluded)
            self.assertFalse(set(excluded) & {row["id"] for row in second["memes"]})
            self.assertEqual(second["catalog"]["total"], 100)
            self.assertEqual(second["catalog"]["remaining"], 94)
            payload = json.loads(complete.call_args.args[2][1]["content"])
            self.assertEqual(len(payload["templates"]), 24)
            with self.assertRaises(models.ModelError):
                models.suggest("barbecue", False, [row["id"] for row in catalog["templates"]])

    def test_http_exclusion_contract_and_bad_payload(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), WorkspaceHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        try:
            with patch("server.suggest", return_value={"memes": []}) as suggest, patch("server.missing_dependencies", return_value=[]):
                connection.request("POST", "/api/suggest", json.dumps({"post": "hello", "use_sentiment": False, "exclude_ids": ["1", "2"]}), {"Content-Type": "application/json"})
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                response.read()
                suggest.assert_called_once_with("hello", False, ["1", "2"])
                connection.request("POST", "/api/suggest", json.dumps({"post": "hello", "exclude_ids": [False]}), {"Content-Type": "application/json"})
                response = connection.getresponse()
                self.assertEqual(response.status, 400)
                response.read()
                self.assertEqual(suggest.call_count, 1)
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
