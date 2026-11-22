from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
CATEGORIES = ("smiley face", "cat", "coffee cup", "sun", "flower", "birthday cake")
EMOTIONS = ("joy", "sadness", "anger", "fear", "surprise", "affection", "neutral", "other")
EMOTION_GROUPS = {
    "admiration": "joy", "amusement": "joy", "anger": "anger", "annoyance": "anger",
    "approval": "joy", "caring": "affection", "confusion": "surprise", "curiosity": "surprise",
    "desire": "other", "disappointment": "sadness", "disapproval": "anger", "disgust": "anger",
    "embarrassment": "sadness", "excitement": "joy", "fear": "fear", "gratitude": "affection",
    "grief": "sadness", "joy": "joy", "love": "affection", "nervousness": "fear",
    "optimism": "joy", "pride": "joy", "realization": "surprise", "relief": "joy",
    "remorse": "sadness", "sadness": "sadness", "surprise": "surprise", "neutral": "neutral",
}


def tokenize(text):
    text = re.sub(r"https?://\S+", " url ", text.lower())
    text = re.sub(r"(?<!\w)@\w+", " user ", text)
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[!?]", text)


def encode(text, vocabulary, length=96):
    tokens = [vocabulary.get(word, 1) for word in tokenize(text)[:length]] or [1]
    return tokens + [0] * (length - len(tokens))
