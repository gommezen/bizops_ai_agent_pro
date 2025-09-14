from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

DANISH_STOPWORDS = [
    "og","i","jeg","det","at","en","den","til","er","som","på","de","med","han","af","for","ikke",
    "der","var","mig","sig","men","et","har","om","vi","min","havde","ham","hun","nu","over","da",
    "fra","du","ud","sin","dem","os","op","man","hans","hvor","eller","hvad","skal","selv","her",
    "alle","vil","blev","kunne","ind","når","være","kom","noget","anden","have","hende","mine",
    "alt","meget","sit","sine","vor","mod","disse","hvis","din","nogle","hos","blive","mange","ad",
    "bliver","hendes","været","thi","jer","så"
]

def read_texts(folder: Path) -> list[str]:
    texts = []
    for p in sorted(Path(folder).glob("*.txt")):
        texts.append(p.read_text(encoding="utf-8", errors="ignore"))
    return texts

def extract_themes(texts: list[str], k:int=3, top_terms:int=6) -> list[str]:
    if not texts:
        return []
    vec = TfidfVectorizer(max_df=0.9, min_df=1, ngram_range=(1,2), stop_words=DANISH_STOPWORDS)
    X = vec.fit_transform(texts)
    k = min(k, X.shape[0])
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    _ = km.fit_predict(X) #labels = km.fit_predict(X)
    terms = vec.get_feature_names_out()
    themes = []
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
    return [{"name": n, "current": c, "target": t} for (n,c,t) in candidates]

def run_needs(input_folder="data/interviews", clusters=4, top_terms=8):
    texts = read_texts(input_folder)
    themes = extract_themes(texts, k=clusters, top_terms=top_terms)
    kpis = propose_kpis(texts)
    return {"themes": themes, "kpis": kpis}
