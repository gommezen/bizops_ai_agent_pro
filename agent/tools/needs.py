from collections.abc import Iterable, Sequence
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# Danish stopwords used during TF-IDF; keeps noise words out even if interview texts are Danish.
DANISH_STOPWORDS = [
    "og",
    "i",
    "jeg",
    "det",
    "at",
    "en",
    "den",
    "til",
    "er",
    "som",
    "på",
    "de",
    "med",
    "han",
    "af",
    "for",
    "ikke",
    "der",
    "var",
    "mig",
    "sig",
    "men",
    "et",
    "har",
    "om",
    "vi",
    "min",
    "havde",
    "ham",
    "hun",
    "nu",
    "over",
    "da",
    "fra",
    "du",
    "ud",
    "sin",
    "dem",
    "os",
    "op",
    "man",
    "hans",
    "hvor",
    "eller",
    "hvad",
    "skal",
    "selv",
    "her",
    "alle",
    "vil",
    "blev",
    "kunne",
    "ind",
    "når",
    "være",
    "kom",
    "noget",
    "anden",
    "have",
    "hende",
    "mine",
    "alt",
    "meget",
    "sit",
    "sine",
    "vor",
    "mod",
    "disse",
    "hvis",
    "din",
    "nogle",
    "hos",
    "blive",
    "mange",
    "ad",
    "bliver",
    "hendes",
    "været",
    "thi",
    "jer",
    "så",
]

ENGLISH_STOPWORDS = [
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
    "so",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "by",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "this",
    "that",
    "these",
    "those",
    "as",
    "at",
    "from",
    "we",
    "you",
    "they",
    "our",
    "your",
    "their",
    "i",
    "he",
    "she",
    "them",
    "us",
    "do",
    "does",
    "did",
]

STOPWORDS = list(set(DANISH_STOPWORDS + ENGLISH_STOPWORDS))


def read_texts(folder: Path | str) -> list[str]:
    """Read all .txt files in a folder as UTF-8 strings."""
    folder = Path(folder)
    texts: list[str] = []
    for p in sorted(folder.glob("*.txt")):
        texts.append(p.read_text(encoding="utf-8", errors="ignore"))
    return texts


def extract_themes(texts: list[str], k: int = 3, top_terms: int = 6) -> list[str]:
    """
    Vectorize texts with TF-IDF (unigrams+bigrams), cluster with KMeans,
    and return top keywords per cluster as a comma-joined string.
    """
    if not texts:
        return []
    vec = TfidfVectorizer(
        max_df=0.9,
        min_df=1,
        ngram_range=(1, 2),
        stop_words=STOPWORDS,  # <- combined DA+EN
    )
    X = vec.fit_transform(texts)
    k = min(k, X.shape[0])
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    _ = km.fit_predict(X)
    terms = vec.get_feature_names_out()
    themes: list[str] = []
    for i in range(km.n_clusters):
        centroid = km.cluster_centers_[i]
        top_idx = centroid.argsort()[-top_terms:][::-1]
        keywords = [terms[j] for j in top_idx]
        themes.append(", ".join(keywords))
    return themes


def propose_kpis(texts: list[str]) -> list[dict]:
    """Return simple KPI candidates (placeholder suggestions) in English."""
    candidates = [
        ("Churn rate", "—", "-15% in 12 months"),
        ("NPS", "—", "+10 pts"),
        ("Time to insight", "—", "-30%"),
        ("Report latency", "—", "weekly instead of monthly"),
    ]
    return [{"name": n, "current": c, "target": t} for (n, c, t) in candidates]


# ---------- Theme polishing (from keywords -> human-readable bullets) ----------
def _contains_any(words: list[str], candidates: list[str]) -> bool:
    wl = [w.lower() for w in words]
    return any(c in wl for c in candidates)


def _polish_one_theme(tokens: list[str]) -> dict[str, list[str] | str]:
    t = [w.strip().lower() for w in tokens if w.strip()]

    # Titles: add English synonyms
    if _contains_any(
        t,
        [
            "regnskab",
            "regnskabet",
            "bogholderi",
            "bilag",
            "accounting",
            "bookkeeping",
            "invoice",
            "invoices",
        ],
    ):
        title = "Automating accounting workflows"
    elif _contains_any(
        t, ["model", "forudsige", "forecast", "prognose", "prediction", "predictive", "modeling"]
    ):
        title = "Forecasting and modeling"
    elif _contains_any(
        t,
        [
            "overblik",
            "status",
            "opslag",
            "manuelle",
            "manuel",
            "overview",
            "status",
            "lookup",
            "lookups",
            "manual",
        ],
    ):
        title = "Overview and efficiency"
    elif _contains_any(
        t,
        [
            "ledelsen",
            "implementeringsplan",
            "standardrapporter",
            "ugentlige",
            "leadership",
            "implementation",
            "standard reports",
            "weekly",
        ],
    ):
        title = "Leadership needs and reporting"
    else:
        title = "Operational improvements"

    bullets: list[str] = []

    # Bullets: add English synonyms
    if _contains_any(t, ["manuel", "manuelle", "fejl", "manual", "error", "errors"]):
        bullets.append("Repeated manual work increases error risk and should be automated.")
    if _contains_any(t, ["regnskab", "regnskabet", "accounting", "invoices"]):
        bullets.append("Automating accounting can free up time for analysis.")
    if _contains_any(
        t, ["standardrapporter", "dag", "ugentlige", "standard reports", "daily", "weekly"]
    ):
        bullets.append("Standard reports on a fixed cadence are requested (daily/weekly).")
    if _contains_any(t, ["forudsige", "model", "forecast", "prediction", "modeling"]):
        bullets.append("Better models are needed to forecast developments.")
    if _contains_any(
        t, ["overblik", "status", "opslag", "overview", "status", "lookup", "lookups"]
    ):
        bullets.append("Lack of a consolidated overview; status lookups take too long.")
    if _contains_any(t, ["ledelsen", "implementeringsplan", "leadership", "implementation"]):
        bullets.append("Leadership requests a phased implementation plan for automation.")
    if not bullets:
        bullets.append("Potential for efficiency gains and better decision support.")

    # De-duplicate while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for b in bullets:
        if b not in seen:
            seen.add(b)
            uniq.append(b)
    bullets = uniq

    return {"title": title, "bullets": bullets, "keywords": tokens}


def _to_tokens(theme_item: Iterable[str] | str) -> list[str]:
    """Accept either '\"a, b, c\"' or ['a','b','c'] and return a token list."""
    if isinstance(theme_item, str):
        return [tok.strip() for tok in theme_item.split(",") if tok.strip()]
    return [str(tok).strip() for tok in theme_item if str(tok).strip()]


def polish_themes(
    raw_themes: Sequence[str | Iterable[str]],
) -> list[dict[str, list[str] | str]]:
    """Convert raw comma-joined keyword strings (or lists) into titled, readable themes."""
    token_lists = [_to_tokens(item) for item in raw_themes]
    return [_polish_one_theme(tokens) for tokens in token_lists]


# --------------------------------------------------------------------


def run_needs(
    input_folder: str | Path = "data/interviews",
    clusters: int = 4,
    top_terms: int = 8,
) -> dict:
    """
    Run the needs analysis and return both raw and polished themes + KPI suggestions.
    - themes: raw comma-joined keywords per cluster
    - themes_polished: list of dicts {title, bullets, keywords}
    """
    folder = Path(input_folder)
    texts = read_texts(folder)
    themes = extract_themes(texts, k=clusters, top_terms=top_terms)
    kpis = propose_kpis(texts)
    polished = polish_themes(themes)

    return {
        "themes": themes,  # backward-compatible key (comma-joined str per cluster)
        "themes_raw": themes,
        "themes_polished": polished,  # [{title, bullets, keywords}, ...]
        "kpis": kpis,
        "k": clusters,
        "top_terms": top_terms,
        "source_dir": str(folder),
    }
