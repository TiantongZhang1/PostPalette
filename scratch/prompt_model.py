"""Randomly initialized GRU encoder/decoder with attention and a copy distribution."""

import re

import torch
from torch import nn

PAD, UNK, BOS, EOS = 0, 1, 2, 3
LENGTH = 40


def words(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[.,!?]", text.lower())


def encode_prompt(text, vocabulary):
    tokens = words(text)[:LENGTH - 1]
    ids = [vocabulary.get(word, UNK) for word in tokens] + [EOS]
    return ids + [PAD] * (LENGTH - len(ids))


class PromptWriter(nn.Module):
    def __init__(self, vocabulary_size, embedding=64, hidden=96):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding, padding_idx=PAD)
        self.encoder = nn.GRU(embedding, hidden, batch_first=True)
        self.decoder = nn.GRU(embedding, hidden, batch_first=True)
        self.output = nn.Linear(hidden * 2, vocabulary_size)
        self.gate = nn.Linear(hidden * 2 + embedding, 1)

    def encode(self, source):
        encoded, _ = self.encoder(self.embedding(source))
        lengths = source.ne(PAD).sum(1) - 1
        state = encoded[torch.arange(source.shape[0], device=source.device), lengths].unsqueeze(0)
        return encoded, state

    def decode(self, source, encoded, inputs, state):
        embedded = self.embedding(inputs)
        decoded, state = self.decoder(embedded, state)
        energy = torch.bmm(decoded, encoded.transpose(1, 2)) / decoded.shape[-1] ** 0.5
        attention = energy.masked_fill(source[:, None, :].eq(PAD), -1e4).softmax(-1)
        context = torch.bmm(attention, encoded)
        features = torch.cat((decoded, context), -1)
        gate = self.gate(torch.cat((features, embedded), -1)).sigmoid()
        probability = self.output(features).softmax(-1) * gate
        probability = probability.scatter_add(2, source[:, None, :].expand(-1, inputs.shape[1], -1), attention * (1 - gate))
        return probability.clamp_min(1e-9).log(), state

    def forward(self, source, inputs):
        encoded, state = self.encode(source)
        return self.decode(source, encoded, inputs, state)[0]

    @torch.inference_mode()
    def write(self, text, vocabulary):
        source = torch.tensor([encode_prompt(text, vocabulary)])
        encoded, state = self.encode(source)
        current = torch.tensor([[BOS]])
        inverse = {value: key for key, value in vocabulary.items()}
        output = []
        trigrams = set()
        for _ in range(LENGTH - 1):
            scores, state = self.decode(source, encoded, current, state)
            scores[0, 0, [PAD, BOS]] = -1e4
            token = int(scores[0, -1].argmax())
            if token == EOS:
                break
            if token == UNK:
                # The shared UNK bucket has no reliable identity. Do not invent a word.
                word = ""
            else:
                word = inverse.get(token, "")
            if word:
                # Explicit decoding guard against a small model looping forever.
                trigram = tuple(output[-2:] + [word])
                if len(output) >= 2:
                    if trigram in trigrams:
                        break
                    trigrams.add(trigram)
                output.append(word)
            current = torch.tensor([[token]])
        sentence = " ".join(output)
        sentence = re.sub(r"\s+([.,!?])", r"\1", sentence).strip()
        return sentence
