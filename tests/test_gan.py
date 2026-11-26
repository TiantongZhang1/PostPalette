"""CPU-only tests for the conditional GAN doodle generator."""

import unittest

import torch

from scratch.networks import ConditionalGAN, create_network
from scratch.train import recognizability, resolve_device


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


if __name__ == "__main__":
    unittest.main()
