"""Retrieve a public meme catalog and shortlist templates for local reranking."""

from functools import lru_cache
import json
import re
from time import monotonic
from urllib.request import Request, urlopen

CATALOG_URL = "https://api.imgflip.com/get_memes"

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


@lru_cache(maxsize=1)
def popular_templates(_time_bucket):
    try:
        request = Request(CATALOG_URL, headers={"User-Agent": "PostPalette/1.0", "Accept": "application/json"})
        with urlopen(request, timeout=4) as response:
            data = json.loads(response.read(262144))
        rows = data["data"]["memes"]
        if data["success"] is True and isinstance(rows, list):
            return {row["id"]: (rank, row["url"]) for rank, row in enumerate(rows)
                    if isinstance(row, dict) and isinstance(row.get("id"), str)
                    and isinstance(row.get("url"), str)
                    and re.fullmatch(r"https://i\.imgflip\.com/[a-z0-9]+\.(?:jpg|jpeg|png|webp)", row["url"])}
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return {}


def shortlist(post):
    popular = popular_templates(int(monotonic() // 600))
    words = set(re.findall(r"[a-z]+", post.lower()))
    candidates = sorted(
        TEMPLATES,
        key=lambda meme: (-len(words.intersection(meme[4].split())), popular.get(meme[0], (999, None))[0]),
    )[:8]
    return [
        {"name": name, "use": use, "image_url": popular.get(meme_id, (999, f"https://i.imgflip.com/{filename}"))[1],
         "source_url": f"https://imgflip.com/memegenerator/{meme_id}"}
        for meme_id, name, filename, use, _ in candidates
    ]
