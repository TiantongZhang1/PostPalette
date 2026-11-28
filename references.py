"""Small local TF-IDF baseline. No trained embeddings or internet search."""

from collections import Counter
from functools import cache
import json
import math
from pathlib import Path
import re


def terms(text):
    english = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
    stop = {"a", "an", "the", "i", "my", "we", "to", "in", "on", "at", "and", "with", "today", "was", "is", "went", "it", "of", "for"}
    output = [word for word in english if word not in stop]
    for segment in re.findall(r"[\u3400-\u9fff]+", text):
        output.extend(segment[i:i + n] for n in (2, 3) for i in range(len(segment) - n + 1)
                      if segment[i:i + n] not in {"今天", "一起", "自己", "朋友", "一下", "可以", "一个", "时间"})
    return Counter(output)


@cache
def index():
    path = Path(__file__).resolve().parent / "content" / "reference_posts.json"
    corpus = json.loads(path.read_text(encoding="utf-8"))
    rows = corpus["items"]
    documents = [terms(row["topic"] + " " + row["text"] + " " + row["keywords"]) for row in rows]
    frequency = Counter(term for doc in documents for term in doc)
    idf = {term: math.log((len(rows) + 1) / (count + 1)) + 1 for term, count in frequency.items()}
    return corpus, documents, idf


def vector(counts, idf):
    values = {term: (1 + math.log(count)) * idf[term] for term, count in counts.items() if term in idf}
    norm = math.sqrt(sum(value * value for value in values.values()))
    return {term: value / norm for term, value in values.items()} if norm else {}


def find_references(post):
    corpus, documents, idf = index()
    query = vector(terms(post), idf)
    matches = []
    for row, counts in zip(corpus["items"], documents):
        document = vector(counts, idf)
        shared = sorted(query.keys() & document.keys(), key=lambda term: query[term] * document[term], reverse=True)
        score = sum(query[term] * document[term] for term in shared)
        if score >= 0.075:
            matches.append({"id": row["id"], "topic": row["topic"], "text": row["text"],
                            "score": round(score, 4), "matched_terms": shared[:4],
                            "source": "AI 编写的项目示例 · 非真实用户帖子"})
    matches.sort(key=lambda row: (-row["score"], row["id"]))
    return {"matches": matches[:3], "method": "TF-IDF cosine (lexical baseline)",
            "corpus_count": len(documents), "corpus_description": corpus["description"]}
