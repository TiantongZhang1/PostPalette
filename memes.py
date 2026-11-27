"""Retrieve a public meme catalog and shortlist templates for local reranking."""

from hashlib import sha256
import json
from pathlib import Path
import re
from secrets import randbits
from threading import Lock
from time import monotonic
from urllib.request import Request, urlopen

CATALOG_URL = "https://api.imgflip.com/get_memes"
ROOT = Path(__file__).resolve().parent
CACHE_FILE = ROOT / ".model-cache" / "memes" / "catalog.json"
SNAPSHOT_FILE = ROOT / "content" / "meme_catalog.json"
ANNOTATIONS_FILE = ROOT / "content" / "meme_annotations.json"
_catalog_lock = Lock()
_catalog_state = None
_catalog_until = 0

# Verified templates and their common uses, kept as a fallback if Imgflip is unreachable.
# (Imgflip ID, title, image filename, use, retrieval terms)
TEMPLATES = (
    ("181913649", "Drake Hotline Bling", "30b1gx.jpg", "reject one option and prefer another", "prefer choose love hate instead better"),
    ("87743020", "Two Buttons", "1g8my4.jpg", "two hard choices", "choose choice decide dilemma either options"),
    ("112126428", "Distracted Boyfriend", "1ur9b0.jpg", "tempted by a new distraction", "distraction distracted procrastinate shiny instead focus"),
    ("217743513", "UNO Draw 25 Cards", "3lmzyx.jpg", "stubborn refusal to do something", "refuse stubborn avoid rather never homework"),
    ("124822590", "Left Exit 12 Off Ramp", "22bdq6.jpg", "abandoning the plan at the last second", "pivot change deadline instead procrastinate plan"),
    ("322841258", "Anakin Padme 4 Panel", "5c7lwq.png", "a reassuring assumption turns out to be wrong", "right surprise awkward suspicious misunderstanding"),
    ("131940431", "Gru's Plan", "26jxvz.jpg", "a plan unexpectedly backfires", "plan fail mistake backfire realize project"),
    ("135256802", "Epic Handshake", "28j0te.jpg", "different groups have a shared experience", "friends together team same agree both collaboration"),
    ("97984", "Disaster Girl", "23ls.jpg", "mischief or satisfaction amid chaos", "chaos disaster trouble prank mischievous"),
    ("4087833", "Waiting Skeleton", "2fm6x.jpg", "a painfully long wait", "wait waiting delay forever late loading queue"),
    ("80707627", "Sad Pablo Escobar", "1c1uej.jpg", "loneliness and disappointment", "sad lonely miss disappointed loss waiting goodbye"),
    ("309868304", "Trade Offer", "54hjww.jpg", "an amusing deal or exchange", "trade deal exchange offer receive"),
    ("188390779", "Woman Yelling At Cat", "345v97.jpg", "an angry complaint meets an unimpressed cat", "cat pet argument angry yelling blame annoyed"),
    ("61579", "One Does Not Simply", "1bij.jpg", "something is much harder than it sounds", "hard difficult struggle work impossible deadline"),
    ("100777631", "Is This A Pigeon", "1o00in.jpg", "confidently misidentifying something", "confused confusion mistake clueless identify"),
    ("28251713", "Oprah You Get A", "gtj5t.jpg", "enthusiastically giving everyone something", "celebrate celebration party congratulations gift reward win"),
    ("505705955", "Absolute Cinema", "8d317n.png", "applauding an impressive achievement", "graduate graduated graduation thesis submitted finish finished proud celebrate success amazing win"),
    ("3218037", "This Is Where I'd Put My Trophy If I Had One", "1wz1x.jpg", "lamenting a missing achievement or reward", "trophy prize lost reward disappointed award"),
    ("247375501", "Buff Doge vs. Cheems", "43a45p.png", "a funny comparison between strong and weak versions", "dog pet workout strength comparing past present"),
    ("93895088", "Expanding Brain", "1jwhww.jpg", "escalating clever or absurd ideas", "idea genius study learn creative evolve"),
    ("55311130", "This Is Fine", "wxica.jpg", "acting calm in a stressful or chaotic situation", "stress stressful tired exhausted deadline chaos panic sleepless overwhelmed"),
    ("178591752", "Tuxedo Winnie The Pooh", "2ybua0.png", "a playful ordinary versus fancy comparison", "fancy elegant formal better rich upgraded"),
)


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _valid_rows(data):
    """Accept only bounded template metadata and image URLs on the existing host."""
    rows = data.get("memes", []) if isinstance(data, dict) else []
    if not isinstance(rows, list):
        return []
    valid = {}
    for row in rows[:1000]:
        if not isinstance(row, dict):
            continue
        meme_id, name, url = row.get("id"), row.get("name"), row.get("url")
        if (isinstance(meme_id, str) and re.fullmatch(r"[0-9]{1,20}", meme_id)
                and isinstance(name, str) and 0 < len(name.strip()) <= 160
                and isinstance(url, str)
                and re.fullmatch(r"https://i\.imgflip\.com/[a-z0-9]+\.(?:jpg|jpeg|png|webp)", url)):
            valid.setdefault(meme_id, {"id": meme_id, "name": name.strip(), "url": url})
    return list(valid.values())


def load_catalog():
    """Merge the full live directory, last good cache, and bundled snapshot.

    Only public template metadata is cached; user posts never leave the machine.
    Successful fetches have a ten-minute TTL; failures retry after one minute.
    """
    global _catalog_state, _catalog_until
    with _catalog_lock:
        if _catalog_state is not None and monotonic() < _catalog_until:
            return _catalog_state
        bundled = _valid_rows(_read_json(SNAPSHOT_FILE))
        cached = _valid_rows(_read_json(CACHE_FILE))
        live = []
        try:
            request = Request(CATALOG_URL, headers={"User-Agent": "PostPalette/1.0", "Accept": "application/json"})
            with urlopen(request, timeout=4) as response:
                data = json.loads(response.read(1048576))
            if isinstance(data, dict) and data.get("success") is True:
                live = _valid_rows(data.get("data"))
        except (OSError, ValueError, TypeError):
            pass
        mode = "live" if live else "cached" if cached else "bundled" if bundled else "fallback"
        merged = {}
        for row in live + cached + bundled:
            merged.setdefault(row["id"], row)
        for meme_id, name, filename, _, _ in TEMPLATES:
            merged.setdefault(meme_id, {"id": meme_id, "name": name, "url": f"https://i.imgflip.com/{filename}"})
        # Retain past popular templates as well as today's list, bounded to 500.
        rows = list(merged.values())[:500]
        if live:
            try:
                CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
                temporary = CACHE_FILE.with_suffix(".tmp")
                temporary.write_text(json.dumps({"memes": rows}, ensure_ascii=False), encoding="utf-8")
                temporary.replace(CACHE_FILE)
            except OSError:
                pass  # A read-only cache must not break recommendations.
        annotations = _read_json(ANNOTATIONS_FILE)
        if not isinstance(annotations, dict):
            annotations = {}
        defaults = {row[0]: {"use": row[3], "terms": row[4]} for row in TEMPLATES}
        templates = []
        for rank, row in enumerate(rows):
            note = annotations.get(row["id"], defaults.get(row["id"], {}))
            templates.append({"id": row["id"], "name": row["name"], "image_url": row["url"],
                              "source_url": f"https://imgflip.com/memegenerator/{row['id']}",
                              "use": note.get("use", "Use only when the template title clearly fits the post."),
                              "terms": note.get("terms", ""), "rank": rank})
        _catalog_state = {"templates": templates, "mode": mode, "live_count": len(live)}
        _catalog_until = monotonic() + (600 if live else 60)
        return _catalog_state


def validate_exclusions(value):
    if (not isinstance(value, list) or len(value) > 120
            or any(not isinstance(item, str) or not re.fullmatch(r"[0-9]{1,20}", item) for item in value)):
        raise ValueError("已看梗图列表必须是最多 120 个有效模板编号。")
    return list(dict.fromkeys(value))


# Explicit lexical expansion, not an embedding model or a learned classifier.
TOPICS = (
    "food eat eating meal barbecue bbq grill cook cooking hungry restaurant dinner lunch snack delicious",
    "friend friends friendship together group team party hangout collaboration",
    "work working job office boss meeting salary paid money pay",
    "study studying homework exam exams school university thesis graduation graduate project deadline",
    "happy joy excited celebrate celebration win success finished finish proud congratulations cheers",
    "sad sadness lonely loneliness disappointed miss loss cry crying",
    "tired exhausted stress stressful overwhelmed panic chaos sleepless burnout",
    "cat kitten dog puppy pet pets animal",
    "exercise gym workout run running bike cycling sport sports strong strength",
    "wait waiting delay late loading queue patience forever",
    "surprise surprised shock shocked unexpected realize realization",
    "fail failure mistake wrong backfire accident disaster burn burnt burning",
    "choice choices choose decision decide dilemma options prefer instead",
    "computer laptop coding code programming bug debugging technology internet wifi",
)
STOP_WORDS = set("a an the i me my we our you your it is are was were be been to of for with and or at in on that this today just so very not have has had do did from".split())


def _words(text):
    words = set(re.findall(r"[a-z]+", text.lower())) - STOP_WORDS
    # A lightweight plural normalization helps title and purpose keywords align.
    return words | {word[:-1] for word in words if len(word) > 4 and word.endswith("s")}


def shortlist(post, exclude_ids=(), *, catalog=None, limit=24):
    catalog = load_catalog() if catalog is None else catalog
    excluded = set(exclude_ids)
    available = [row for row in catalog["templates"] if row["id"] not in excluded]
    words = _words(post)
    expanded = set(words)
    for topic in TOPICS:
        terms = set(topic.split())
        if words & terms:
            expanded.update(terms)
    seed = str(randbits(64))

    def relevance(row):
        terms = _words(row["name"] + " " + row["use"] + " " + row["terms"])
        return 3 * len(words & terms) + len((expanded - words) & terms)

    # Relevance first, varied tie breaks. Popularity is only a small secondary hint.
    ranked = sorted(available, key=lambda row: (
        -relevance(row), row["rank"] // 25,
        sha256((seed + row["id"]).encode()).digest()))
    # Keep 18 lexical matches and explore six from the next 36 for LLM reranking.
    primary = ranked[:max(0, limit - 6)]
    rest = ranked[len(primary):len(primary) + 36]
    rest.sort(key=lambda row: sha256((row["id"] + seed).encode()).digest())
    return primary + rest[:limit - len(primary)]
