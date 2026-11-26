"""CPU-only tests for the conditional GAN doodle generator."""

from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
import pathlib
import tempfile
import threading
import unittest
from unittest.mock import patch

import torch

from scratch.networks import ConditionalGAN, create_network
from scratch.train import recognizability, resolve_device
from server import WorkspaceHandler

PNG = b"\x89PNG\r\n\x1a\nfixture"
DOODLE = {"category": "cat", "color": "#25203c", "mode": "cover", "seed": 1}


class ConditionalGANTests(unittest.TestCase):
    def test_generate_shape_and_ink_range(self):
        model = ConditionalGAN().eval()
        with torch.inference_mode():
            images = model.generate(torch.randn(4, model.latent), torch.tensor([0, 2, 5, 5]))
        self.assertEqual(images.shape, (4, 1, 28, 28))
        self.assertGreaterEqual(float(images.min()), 0.0)
        self.assertLessEqual(float(images.max()), 1.0)

    def test_generate_is_conditioned_on_label(self):
        torch.manual_seed(2026)
        model = ConditionalGAN().eval()
        noise = torch.randn(1, model.latent).expand(2, -1)
        with torch.inference_mode():
            images = model.generate(noise, torch.tensor([0, 5]))
        self.assertFalse(torch.allclose(images[0], images[1]))

    def test_same_seed_gives_the_same_drawing(self):
        model = ConditionalGAN().eval()
        labels = torch.tensor([3, 3])
        with torch.inference_mode():
            first = model.generate(torch.randn(2, model.latent, generator=torch.Generator().manual_seed(7)), labels)
            second = model.generate(torch.randn(2, model.latent, generator=torch.Generator().manual_seed(7)), labels)
        self.assertTrue(torch.equal(first, second))

    def test_discriminate_returns_real_and_class_logits(self):
        model = ConditionalGAN().eval()
        with torch.inference_mode():
            real, classes = model.discriminate(torch.rand(3, 1, 28, 28))
        self.assertEqual(real.shape, (3,))
        self.assertEqual(classes.shape, (3, 6))

    def test_factory_builds_gan(self):
        self.assertIsInstance(create_network("gan", {}), ConditionalGAN)


class TrainingSupportTests(unittest.TestCase):
    def test_existing_tasks_keep_their_cuda_default(self):
        for task in ("emotion", "cnn", "vae"):
            self.assertEqual(resolve_device(task, None), "cuda")

    def test_gan_defaults_to_cpu(self):
        self.assertEqual(resolve_device("gan", None), "cpu")

    def test_explicit_device_always_wins(self):
        self.assertEqual(resolve_device("gan", "cuda"), "cuda")
        self.assertEqual(resolve_device("vae", "cpu"), "cpu")

    def test_recognizability_counts_matching_predictions(self):
        class FixedCNN(torch.nn.Module):
            def forward(self, images):
                return torch.tensor([[9.0, 0, 0, 0, 0, 0], [0, 0, 9.0, 0, 0, 0]])

        score = recognizability(torch.zeros(2, 1, 28, 28), torch.tensor([0, 5]), FixedCNN())
        self.assertAlmostEqual(score, 0.5)


class ApplicationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), WorkspaceHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def post(self, path, body):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            connection.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
            response = connection.getresponse()
            response.read()
            return response.status
        finally:
            connection.close()

    def test_health_lists_the_gan_checkpoint(self):
        from models import training_status
        self.assertIn("gan", training_status())

    def test_unknown_generator_is_rejected_before_the_model_layer(self):
        with patch("server.generate_image") as generate:
            status = self.post("/api/image", {**DOODLE, "generator": "stable-diffusion"})
        self.assertEqual(status, 400)
        generate.assert_not_called()

    def test_known_generator_reaches_the_model_layer(self):
        with patch("server.generate_image", return_value=(PNG, 0.9)) as generate:
            status = self.post("/api/image", {**DOODLE, "generator": "gan"})
        self.assertEqual(status, 200)
        self.assertEqual(generate.call_args.kwargs["generator"], "gan")

    def test_generator_defaults_to_the_vae(self):
        with patch("server.generate_image", return_value=(PNG, 0.9)) as generate:
            status = self.post("/api/image", DOODLE)
        self.assertEqual(status, 200)
        self.assertEqual(generate.call_args.kwargs["generator"], "vae")

    def test_missing_checkpoint_reports_a_model_error(self):
        import models
        with tempfile.TemporaryDirectory() as empty:
            with patch.object(models, "ARTIFACTS", pathlib.Path(empty)):
                with self.assertRaisesRegex(models.ModelError, "gan"):
                    models.generate_image("cat", "#25203c", False, 2026, generator="gan")

    def test_unknown_generator_never_reaches_a_checkpoint(self):
        import models
        with self.assertRaisesRegex(models.ModelError, "生成模型"):
            models.generate_image("cat", "#25203c", False, 2026, generator="stable-diffusion")


if __name__ == "__main__":
    unittest.main()
