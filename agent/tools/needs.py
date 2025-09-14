from collections.abc import Iterable, Sequence
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

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


def read_texts(folder: Path | str) -> list[str]:
    folder = Path(folder)
    texts: list[str] = []
    for p in sorted(folder.glob("*.txt")):
        texts.append(p.read_text(encoding="utf-8", errors="ignore"))
    return texts


def extract_themes(texts: list[str], k: int = 3, top_terms: int = 6) -> list[str]:
    if not texts:
        return []
    vec = TfidfVectorizer(max_df=0.9, min_df=1, ngram_range=(1, 2), stop_words=DANISH_STOPWORDS)
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
    candidates = [
        ("Churn-rate", "—", "−15% på 12 mdr"),
        ("NPS", "—", "+10 pts"),
        ("Time-to-Insight", "—", "−30%"),
        ("Rapport-latens", "—", "ugentlig i stedet for månedlig"),
    ]
    return [{"name": n, "current": c, "target": t} for (n, c, t) in candidates]


# ---------- Polering af temaer (fra nøgleord -> sætninger) ----------
def _contains_any(words: list[str], candidates: list[str]) -> bool:
    wl = [w.lower() for w in words]
    return any(c in wl for c in candidates)


def _polish_one_theme(tokens: list[str]) -> dict[str, list[str] | str]:
    t = [w.strip().lower() for w in tokens if w.strip()]

    if _contains_any(t, ["regnskab", "regnskabet", "bogholderi", "bilag"]):
        title = "Automatisering af regnskabsarbejde"
    elif _contains_any(t, ["model", "forudsige", "forecast", "prognose"]):
        title = "Prognoser og modeller"
    elif _contains_any(t, ["overblik", "status", "opslag", "manuelle", "manuel"]):
        title = "Overblik og effektivitet"
    elif _contains_any(t, ["ledelsen", "implementeringsplan", "standardrapporter", "ugentlige"]):
        title = "Ledelsesbehov og rapportering"
    else:
        title = "Operationelle forbedringer"

    bullets: list[str] = []
    if _contains_any(t, ["manuel", "manuelle", "fejl"]):
        bullets.append("Gentagne manuelle opgaver øger risikoen for fejl og bør automatiseres.")
    if _contains_any(t, ["regnskab", "regnskabet"]):
        bullets.append("Automatisering i regnskabet kan frigive tid til analyse.")
    if _contains_any(t, ["standardrapporter", "dag", "ugentlige"]):
        bullets.append("Standardrapporter på fast frekvens efterspørges (dagligt/ugentligt).")
    if _contains_any(t, ["forudsige", "model"]):
        bullets.append("Der ønskes bedre modeller til at forudsige udviklingen.")
    if _contains_any(t, ["overblik", "status", "opslag"]):
        bullets.append("Mangler samlet overblik; status og opslag tager for lang tid.")
    if _contains_any(t, ["ledelsen", "implementeringsplan"]):
        bullets.append("Ledelsen efterspørger en trinvis implementeringsplan for automatisering.")
    if not bullets:
        bullets.append("Potentiale for effektivisering og bedre beslutningsstøtte.")

    # Fjern dubletter, bevar rækkefølge
    seen: set[str] = set()
    uniq: list[str] = []
    for b in bullets:
        if b not in seen:
            seen.add(b)
            uniq.append(b)
    bullets = uniq

    return {"title": title, "bullets": bullets, "keywords": tokens}


def _to_tokens(theme_item: Iterable[str] | str) -> list[str]:
    """Accepter både '\"a, b, c\"' og ['a','b','c'] og returnér tokens-listen."""
    if isinstance(theme_item, str):
        return [tok.strip() for tok in theme_item.split(",") if tok.strip()]
    return [str(tok).strip() for tok in theme_item if str(tok).strip()]


def polish_themes(
    raw_themes: Sequence[str | Iterable[str]],
) -> list[dict[str, list[str] | str]]:
    token_lists = [_to_tokens(item) for item in raw_themes]
    return [_polish_one_theme(tokens) for tokens in token_lists]


# --------------------------------------------------------------------


def run_needs(
    input_folder: str | Path = "data/interviews",
    clusters: int = 4,
    top_terms: int = 8,
) -> dict:
    """Kør behovsanalysen og returnér både rå og polerede temaer + KPI-forslag."""
    folder = Path(input_folder)
    texts = read_texts(folder)
    themes = extract_themes(texts, k=clusters, top_terms=top_terms)
    kpis = propose_kpis(texts)
    polished = polish_themes(themes)

    return {
        "themes": themes,  # bagudkompatibel nøgle (komma-separeret str pr. klynge)
        "themes_raw": themes,
        "themes_polished": polished,  # [{title, bullets, keywords}, ...]
        "kpis": kpis,
        "k": clusters,
        "top_terms": top_terms,
        "source_dir": str(folder),
    }
