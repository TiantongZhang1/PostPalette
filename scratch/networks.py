"""All weights in these three networks start with random initialization."""

import torch
from torch import nn
from torch.nn import functional as F


class EmotionEncoder(nn.Module):
    def __init__(self, vocab_size, classes=8, length=96, width=128, layers=2, heads=4, dropout=0.1):
        super().__init__()
        self.words = nn.Embedding(vocab_size, width, padding_idx=0)
        self.positions = nn.Embedding(length, width)
        block = nn.TransformerEncoderLayer(
            width, heads, dim_feedforward=width * 2, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        # PyTorch clones the block; independently initialize each layer.
        for layer in self.encoder.layers:
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            nn.init.xavier_uniform_(layer.self_attn.out_proj.weight)
            nn.init.xavier_uniform_(layer.linear1.weight)
            nn.init.xavier_uniform_(layer.linear2.weight)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, classes))

    def forward(self, tokens):
        mask = tokens.eq(0)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.encoder(self.words(tokens) + self.positions(positions), src_key_padding_mask=mask)
        valid = (~mask).unsqueeze(-1)
        pooled = (hidden * valid).sum(1) / valid.sum(1).clamp_min(1)
        return self.head(pooled)


class DoodleCNN(nn.Module):
    def __init__(self, classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(48, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(128, classes),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


class ConditionalVAE(nn.Module):
    def __init__(self, classes=6, latent=32):
        super().__init__()
        self.classes = classes
        self.latent = latent
        self.encoder = nn.Sequential(nn.Linear(784 + classes, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.mean = nn.Linear(256, latent)
        self.log_variance = nn.Linear(256, latent)
        self.decoder = nn.Sequential(
            nn.Linear(latent + classes, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 784),
        )

    def decode(self, latent, labels):
        condition = F.one_hot(labels, self.classes).to(latent.dtype)
        return self.decoder(torch.cat((latent, condition), dim=1)).reshape(-1, 1, 28, 28)

    def forward(self, images, labels, sample=True):
        condition = F.one_hot(labels, self.classes).to(images.dtype)
        hidden = self.encoder(torch.cat((images.flatten(1), condition), dim=1))
        mean, log_variance = self.mean(hidden), self.log_variance(hidden).clamp(-12, 12)
        latent = mean + torch.randn_like(mean) * (0.5 * log_variance).exp() if sample else mean
        return self.decode(latent, labels), mean, log_variance


def vae_loss(logits, images, mean, log_variance, beta):
    reconstruction = F.binary_cross_entropy_with_logits(logits, images, reduction="none").flatten(1).sum(1).mean()
    divergence = -0.5 * (1 + log_variance - mean.square() - log_variance.exp()).sum(1).mean()
    return (reconstruction + beta * divergence) / 784, reconstruction, divergence


def create_network(task, config):
    return {"emotion": EmotionEncoder, "cnn": DoodleCNN, "vae": ConditionalVAE}[task](**config)
