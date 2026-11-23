"""Random RGB diffusion weights; language features can come from the permitted Qwen LLM."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def configure_cuda():
    # Conservative attention kernels for this Windows/Blackwell CUDA stack.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ResidualBlock(nn.Module):
    def __init__(self, incoming, outgoing, context):
        super().__init__()
        self.first = nn.Sequential(nn.GroupNorm(8, incoming), nn.SiLU(), nn.Conv2d(incoming, outgoing, 3, padding=1))
        self.condition = nn.Linear(context, outgoing * 2)
        self.norm = nn.GroupNorm(8, outgoing)
        self.last = nn.Sequential(nn.SiLU(), nn.Dropout(0.1), nn.Conv2d(outgoing, outgoing, 3, padding=1))
        self.skip = nn.Conv2d(incoming, outgoing, 1) if incoming != outgoing else nn.Identity()
        nn.init.zeros_(self.last[-1].weight)
        nn.init.zeros_(self.last[-1].bias)

    def forward(self, image, context):
        hidden = self.first(image)
        scale, shift = self.condition(context).chunk(2, dim=1)
        hidden = self.norm(hidden) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return self.skip(image) + self.last(hidden)


class Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.output = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, image, text=None, valid=None):
        batch, channels, height, width = image.shape
        q, k, v = self.qkv(self.norm(image)).reshape(batch, 3, 4, channels // 4, height * width).unbind(1)
        result = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return image + self.output(result.transpose(-1, -2).reshape(batch, channels, height, width))


class CrossAttention(nn.Module):
    def __init__(self, channels, text_width):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.query = nn.Conv2d(channels, channels, 1)
        self.key_value = nn.Linear(text_width, channels * 2)
        self.output = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, image, text, valid):
        batch, channels, height, width = image.shape
        q = self.query(self.norm(image)).flatten(2).reshape(batch, 4, channels // 4, -1).transpose(-1, -2)
        k, v = self.key_value(text).reshape(batch, text.shape[1], 2, 4, channels // 4).permute(2, 0, 3, 1, 4).unbind(0)
        hidden = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :])
        hidden = hidden.transpose(-1, -2).reshape(batch, channels, height, width)
        return image + self.output(hidden)


class PhotoUNet(nn.Module):
    def __init__(self, vocab_size, width=48, length=48, feature_dim=0):
        super().__init__()
        self.width, self.length = width, length
        self.feature_dim = feature_dim
        context = width * 4
        self.time = nn.Sequential(nn.Linear(width, context), nn.SiLU(), nn.Linear(context, context))
        self.words = nn.Embedding(vocab_size, context, padding_idx=0)
        self.language_projection = nn.Linear(feature_dim, context) if feature_dim else None
        self.positions = nn.Embedding(length, context)
        encoder = nn.TransformerEncoderLayer(
            context, 4, context * 2, dropout=0.1, batch_first=True, norm_first=True, activation="gelu",
        )
        self.text_encoder = nn.TransformerEncoder(encoder, 2, enable_nested_tensor=False)
        for layer in self.text_encoder.layers:
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            nn.init.xavier_uniform_(layer.linear1.weight)
            nn.init.xavier_uniform_(layer.linear2.weight)
        self.text_norm = nn.LayerNorm(context)
        self.pooled = nn.Linear(context, context)
        self.input = nn.Conv2d(3, width, 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        incoming = width
        sizes = (width, width * 2, width * 3, width * 4)
        for index, outgoing in enumerate(sizes):
            self.down_blocks.append(nn.ModuleList([
                ResidualBlock(incoming, outgoing, context), ResidualBlock(outgoing, outgoing, context),
                nn.ModuleList([Attention(outgoing), CrossAttention(outgoing, context)]) if index >= 2 else nn.ModuleList(),
            ]))
            if index < 3:
                self.downsample.append(nn.Conv2d(outgoing, outgoing, 3, stride=2, padding=1))
            incoming = outgoing
        self.middle = nn.ModuleList([
            ResidualBlock(incoming, incoming, context), Attention(incoming), ResidualBlock(incoming, incoming, context),
        ])
        self.up_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for index, outgoing in enumerate(reversed(sizes)):
            self.up_blocks.append(nn.ModuleList([
                ResidualBlock(incoming + outgoing, outgoing, context), ResidualBlock(outgoing, outgoing, context),
                nn.ModuleList([Attention(outgoing), CrossAttention(outgoing, context)]) if index < 2 else nn.ModuleList(),
            ]))
            if index < 3:
                self.upsample.append(nn.Conv2d(outgoing, outgoing, 3, padding=1))
            incoming = outgoing
        self.output = nn.Sequential(nn.GroupNorm(8, width), nn.SiLU(), nn.Conv2d(width, 3, 3, padding=1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def encode_text(self, tokens):
        valid = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device) if self.feature_dim else tokens.ne(0)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        embedded = self.language_projection(tokens.float()) if self.feature_dim else self.words(tokens)
        hidden = self.text_encoder(embedded + self.positions(positions), src_key_padding_mask=~valid)
        hidden = self.text_norm(hidden)
        pooled = (hidden * valid[:, :, None]).sum(1) / valid.sum(1).clamp_min(1)[:, None]
        return hidden, valid, self.pooled(pooled)

    def forward(self, images, times, tokens, text_context=None):
        text, valid, pooled = self.encode_text(tokens) if text_context is None else text_context
        half = self.width // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=images.device) / (half - 1))
        phases = times.float()[:, None] * frequencies[None]
        context = self.time(torch.cat((phases.sin(), phases.cos()), dim=1)) + pooled
        hidden = self.input(images)
        skips = []
        for index, (first, second, attention) in enumerate(self.down_blocks):
            hidden = second(first(hidden, context), context)
            for layer in attention:
                hidden = layer(hidden, text, valid)
            skips.append(hidden)
            if index < 3:
                hidden = self.downsample[index](hidden)
        first, attention, second = self.middle
        hidden = second(attention(first(hidden, context)), context)
        for index, (first, second, attention) in enumerate(self.up_blocks):
            hidden = second(first(torch.cat((hidden, skips.pop()), dim=1), context), context)
            for layer in attention:
                hidden = layer(hidden, text, valid)
            if index < 3:
                hidden = self.upsample[index](F.interpolate(hidden, scale_factor=2, mode="nearest"))
        return self.output(hidden)


def noise_schedule(device, steps=1000):
    times = torch.linspace(0, 1, steps + 1, device=device, dtype=torch.float64)
    cumulative = torch.cos((times + 0.008) / 1.008 * math.pi / 2).square()
    cumulative = cumulative / cumulative[0]
    beta = (1 - cumulative[1:] / cumulative[:-1]).clamp(max=0.999)
    return (1 - beta).cumprod(0).float()


@torch.inference_mode()
def sample(model, tokens, size=64, steps=1000, seed=2026, guidance=2.0, same_noise=False, method="ddpm"):
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    noise_shape = (1 if same_noise else len(tokens), 3, size, size)
    images = torch.randn(noise_shape, device=device, generator=generator)
    if same_noise:
        images = images.expand(len(tokens), -1, -1, -1).clone()
    alphas = noise_schedule(device)
    if method == "ddpm":
        times = torch.arange(999, -1, -1, device=device)
    elif method == "ddim":
        times = torch.linspace(999, 0, steps, device=device).round().long()
    else:
        raise ValueError("Expected ddpm or ddim sampling")
    null = torch.zeros_like(tokens)
    if not model.feature_dim:
        null[:, 0] = 2
    model.eval()
    text_context = model.encode_text(torch.cat((null, tokens)))
    for index, timestep in enumerate(times):
        t = timestep.expand(len(tokens))
        both = model(torch.cat((images, images)), torch.cat((t, t)), None, text_context=text_context)
        unconditional, conditional = both.chunk(2)
        prediction = unconditional + guidance * (conditional - unconditional)
        a = alphas[timestep]
        previous = alphas[times[index + 1]] if index + 1 < len(times) else torch.ones((), device=device)
        clean = ((images - (1 - a).sqrt() * prediction) / a.sqrt()).clamp(-1, 1)
        if method == "ddpm":
            alpha = a / previous
            beta = 1 - alpha
            mean = beta * previous.sqrt() / (1 - a) * clean + (1 - previous) * alpha.sqrt() / (1 - a) * images
            if index + 1 < len(times):
                variance = (beta * (1 - previous) / (1 - a)).clamp_min(0)
                noise = torch.randn(noise_shape, device=device, generator=generator)
                if same_noise:
                    noise = noise.expand_as(images)
                images = mean + variance.sqrt() * noise
            else:
                images = mean
        else:
            prediction = (images - a.sqrt() * clean) / (1 - a).sqrt()
            images = previous.sqrt() * clean + (1 - previous).sqrt() * prediction
    return ((images + 1) / 2).clamp(0, 1)
