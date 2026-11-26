"""Offline tests for models.suggest / generate_prompt / generate_image.

Every trained model, Qwen, and the Imgflip catalog are replaced with in-memory
fakes. No checkpoint files are created or read: missing-weight cases point
models.ARTIFACTS at an empty temporary directory, so results do not depend on
whether a developer has artifacts/ locally.
"""

from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image

import models
from models import ModelError
from scratch.config import CATEGORIES, EMOTIONS

CANDIDATES = [{"name": f"template-{number}", "use": "use", "image_url": "", "source_url": ""}
              for number in range(1, 6)]
VALID_REPLY = '{"meme_numbers":[2,1,3],"category":"cat","caption":"Finally done!"}'


class FakeLanguageModel:
    """Stands in for models.complete: records each chat and returns scripted replies."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, tokenizer, model, messages, max_tokens, sample=False):
        self.calls.append(messages)
        return self.replies.pop(0)


class FakeEmotionClassifier:
    def __call__(self, tokens):
        logits = torch.full((1, len(EMOTIONS)), -4.0)
        logits[0, EMOTIONS.index("joy")] = 3.0
        return logits


class FakePromptWriter:
    def __init__(self):
        self.inputs = []

    def write(self, text, vocabulary):
        self.inputs.append(text)
        return "friends at a grill in a park"


class ArtifactsCase(unittest.TestCase):
    """Point models.ARTIFACTS at an empty directory for the duration of each test."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.artifacts = Path(directory.name)
        patcher = patch.object(models, "ARTIFACTS", self.artifacts)
        patcher.start()
        self.addCleanup(patcher.stop)

    def patch(self, name, **kwargs):
        patcher = patch.object(models, name, **kwargs)
        mocked = patcher.start()
        self.addCleanup(patcher.stop)
        return mocked


class SuggestTests(ArtifactsCase):
    def setUp(self):
        super().setUp()
        self.patch("language_model", return_value=(None, None))
        self.patch("shortlist", return_value=CANDIDATES)

    def test_uses_emotion_model_when_weights_exist(self):
        self.patch("has_checkpoint", return_value=True)
        self.patch("trained_model", return_value=(FakeEmotionClassifier(), {"vocabulary": {"done": 2}}))
        qwen = FakeLanguageModel(VALID_REPLY)
        self.patch("complete", side_effect=qwen)

        result = models.suggest("finally done with my thesis", True)

        self.assertEqual([meme["name"] for meme in result["memes"]], ["template-2", "template-1", "template-3"])
        self.assertEqual(result["category"], "cat")
        self.assertEqual(result["caption"], "Finally done!")
        self.assertEqual(result["sentiment"]["label"], "joy")
        self.assertTrue(result["emotion_used"])
        self.assertFalse(result["degraded"])
        self.assertEqual(result["notices"], [])
        payload = json.loads(qwen.calls[0][1]["content"])
        self.assertEqual(payload["emotion_hint"]["label"], "joy")

    def test_missing_emotion_weights_degrades_to_qwen_only(self):
        loader = self.patch("trained_model")
        self.patch("complete", side_effect=FakeLanguageModel(VALID_REPLY))

        result = models.suggest("finally done with my thesis", True)

        loader.assert_not_called()
        self.assertEqual(len(result["memes"]), 3)
        self.assertIsNone(result["sentiment"])
        self.assertTrue(result["emotion_requested"])
        self.assertFalse(result["emotion_used"])
        self.assertTrue(result["degraded"])
        self.assertIn("artifacts/emotion/best.pt", result["notices"][0])

    def test_sentiment_off_is_not_reported_as_degraded(self):
        self.patch("complete", side_effect=FakeLanguageModel(VALID_REPLY))

        result = models.suggest("finally done", False)

        self.assertFalse(result["emotion_requested"])
        self.assertFalse(result["degraded"])
        self.assertEqual(result["notices"], [])

    def test_invalid_category_keeps_meme_results(self):
        qwen = FakeLanguageModel('{"meme_numbers":[1,2,3],"category":"dog","caption":"Hi"}')
        self.patch("complete", side_effect=qwen)

        result = models.suggest("hello there", False)

        self.assertEqual(len(qwen.calls), 1, "an invalid category alone must not trigger a retry")
        self.assertIsNone(result["category"])
        self.assertEqual(len(result["memes"]), 3)
        self.assertTrue(result["degraded"])
        self.assertTrue(any("涂鸦类别" in notice for notice in result["notices"]))

    def test_invalid_rankings_retry_then_fail(self):
        bad = '{"meme_numbers":[1],"category":"cat","caption":"Hi"}'
        qwen = FakeLanguageModel(bad, bad)
        self.patch("complete", side_effect=qwen)

        with self.assertRaises(ModelError):
            models.suggest("hello there", False)
        self.assertEqual(len(qwen.calls), 2)

    def test_chinese_post_is_translated_before_ranking(self):
        qwen = FakeLanguageModel("We had a barbecue in the park.", VALID_REPLY)
        self.patch("complete", side_effect=qwen)

        models.suggest("今天在公园烧烤", False)

        self.assertEqual(qwen.calls[0][1]["content"], "今天在公园烧烤")
        self.assertEqual(json.loads(qwen.calls[1][1]["content"])["post"], "We had a barbecue in the park.")

    def test_failed_translation_keeps_suggest_message(self):
        self.patch("complete", side_effect=FakeLanguageModel("还是中文"))

        with self.assertRaisesRegex(ModelError, "^无法转换为英文，请调整帖子后重试。$"):
            models.suggest("今天在公园烧烤", False)

    def test_unexpected_failure_becomes_model_error(self):
        self.patch("complete", side_effect=RuntimeError("generation crashed"))

        with self.assertRaises(ModelError) as caught:
            models.suggest("hello there", False)
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)


class ParseSuggestionsTests(unittest.TestCase):
    def test_category_is_optional(self):
        memes, category, caption = models.parse_suggestions(
            'noise {"meme_numbers":[3,1,2],"category":"unicorn","caption":" Yes "} noise', CANDIDATES)
        self.assertEqual([meme["name"] for meme in memes], ["template-3", "template-1", "template-2"])
        self.assertIsNone(category)
        self.assertEqual(caption, "Yes")

    def test_valid_category_is_kept(self):
        _, category, _ = models.parse_suggestions(VALID_REPLY, CANDIDATES)
        self.assertIn(category, CATEGORIES)

    def test_too_few_templates_is_rejected(self):
        with self.assertRaises(ValueError):
            models.parse_suggestions('{"meme_numbers":[1,1,9],"category":"cat","caption":"x"}', CANDIDATES)


class GeneratePromptTests(ArtifactsCase):
    def test_missing_prompt_weights_fail_before_loading_qwen(self):
        qwen = self.patch("language_model")

        with self.assertRaisesRegex(ModelError, "prompt"):
            models.generate_prompt("friends having a barbecue")
        qwen.assert_not_called()

    def test_english_post_skips_translation(self):
        self.patch("require_checkpoint")
        writer = FakePromptWriter()
        self.patch("prompt_writer", return_value=(writer, {"vocabulary": {}, "step": 1200}))
        qwen = self.patch("language_model")

        result = models.generate_prompt("Friends having a barbecue in the park")

        qwen.assert_not_called()
        self.assertEqual(writer.inputs, ["Friends having a barbecue in the park"])
        self.assertFalse(result["translated"])
        self.assertEqual(result["draft"], "friends at a grill in a park")
        self.assertEqual(result["step"], 1200)
        self.assertIn("Original scene (authoritative): Friends having a barbecue in the park", result["prompt"])

    def test_chinese_post_is_translated(self):
        self.patch("require_checkpoint")
        writer = FakePromptWriter()
        self.patch("prompt_writer", return_value=(writer, {"vocabulary": {}, "step": 1200}))
        self.patch("language_model", return_value=(None, None))
        self.patch("complete", side_effect=FakeLanguageModel("We had a barbecue in the park."))

        result = models.generate_prompt("今天在公园烧烤")

        self.assertTrue(result["translated"])
        self.assertEqual(result["source"], "今天在公园烧烤")
        self.assertEqual(writer.inputs, ["We had a barbecue in the park."])

    def test_failed_translation_keeps_prompt_message(self):
        self.patch("require_checkpoint")
        self.patch("language_model", return_value=(None, None))
        self.patch("complete", side_effect=FakeLanguageModel("还是中文"))

        with self.assertRaisesRegex(ModelError, "^中文翻译失败，请改用英文描述。$"):
            models.generate_prompt("今天在公园烧烤")

    def test_qwen_load_failure_becomes_model_error(self):
        self.patch("require_checkpoint")
        self.patch("language_model", side_effect=OSError("hub unreachable"))

        with self.assertRaises(ModelError) as caught:
            models.generate_prompt("今天在公园烧烤")
        self.assertIsInstance(caught.exception.__cause__, OSError)


class FakeVAE:
    latent = 32

    def decode(self, noise, labels):
        # Candidate k is a uniform image with logit k - 8, so later candidates are darker.
        levels = torch.arange(noise.shape[0], dtype=torch.float32) - 8
        return levels.view(-1, 1, 1, 1).expand(-1, 1, 28, 28).clone()


class FakeCNN:
    def __init__(self, preferred):
        self.preferred = preferred

    def __call__(self, drawings):
        logits = torch.zeros(drawings.shape[0], len(CATEGORIES))
        logits[self.preferred, CATEGORIES.index("cat")] = 5.0
        return logits


class GenerateImageTests(ArtifactsCase):
    def test_missing_weights_fail_without_loading_models(self):
        loader = self.patch("trained_model")

        with self.assertRaisesRegex(ModelError, "vae"):
            models.generate_image("cat", "#25203c", False, 2026)
        loader.assert_not_called()

    def test_cnn_selects_candidate_and_sticker_is_transparent(self):
        self.patch("require_checkpoint")
        networks = {"vae": (FakeVAE(), {}), "cnn": (FakeCNN(preferred=12), {})}
        self.patch("trained_model", side_effect=lambda task: networks[task])

        sticker, score = models.generate_image("cat", "#d94f88", True, 2026)
        cover, _ = models.generate_image("cat", "#d94f88", False, 2026)

        with Image.open(BytesIO(sticker)) as image:
            self.assertEqual((image.mode, image.size), ("RGBA", (512, 512)))
            self.assertEqual(image.getpixel((0, 0))[3], 0, "sticker corners must be transparent")
            # Alpha compositing rounds each channel, so allow a small difference.
            for actual, wanted in zip(image.getpixel((256, 256))[:3], (0xD9, 0x4F, 0x88)):
                self.assertLessEqual(abs(actual - wanted), 2)
        with Image.open(BytesIO(cover)) as image:
            self.assertEqual((image.mode, image.size), ("RGB", (512, 512)))
            self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))
        expected = torch.tensor([5.0] + [0.0] * (len(CATEGORIES) - 1)).softmax(0)[0]
        self.assertAlmostEqual(score, float(expected), places=5)

        # The CNN choice decides which candidate is drawn: candidate 0 is almost blank.
        networks["cnn"] = (FakeCNN(preferred=0), {})
        faint, _ = models.generate_image("cat", "#d94f88", True, 2026)
        with Image.open(BytesIO(sticker)) as dark, Image.open(BytesIO(faint)) as light:
            self.assertGreater(dark.getpixel((256, 256))[3], 200)
            self.assertEqual(light.getpixel((256, 256))[3], 0)


if __name__ == "__main__":
    unittest.main()
