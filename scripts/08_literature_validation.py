"""
AIRepurposer - Step 8 (Literature Validation)

✔ Hybrid disease handling (user-defined + auto-inference)
✔ Multi-query PubMed mining
✔ Title-based relevance scoring
✔ Log-normalized evidence
✔ Composite Literature Score (mathematically defined)
✔ Score normalization (0–1)
✔ Evidence classification
✔ Fully reproducible

Author: Ghada Mahjoub 
"""

import requests
import pandas as pd
import json
import time
import numpy as np

# =========================
# 🔹 CONFIG PARAMETERS
# =========================
ALPHA = 0.4   # drug-target weight
BETA  = 0.3   # drug-disease weight
GAMMA = 0.2   # target-disease weight
DELTA = 0.1   # relevance weight

KEYWORDS = [
    "inhibit", "binding", "target", "ic50",
    "affinity", "antitumor", "activity", "block",
    "suppress", "antagonist"
]

# =========================
# 🔹 LOAD CONFIG
# =========================
def load_config():
    with open("project_config.json") as f:
        return json.load(f)

# =========================
# 🔹 DISEASE RESOLUTION
# =========================
def resolve_disease(user_disease=None, target_name=None):

    if user_disease and str(user_disease).strip():
        print(f"🧾 Using user-defined disease: {user_disease}")
        return user_disease.lower()

    if target_name:
        t = target_name.lower()

        if any(k in t for k in ["cancer", "onc", "tumor", "kinase"]):
            inferred = "cancer"
        elif any(k in t for k in ["alzheimer", "amyloid"]):
            inferred = "alzheimer disease"
        elif any(k in t for k in ["diabetes", "glucose"]):
            inferred = "diabetes"
        else:
            inferred = "disease"

        print(f"🤖 Auto-inferred disease: {inferred}")
        return inferred

    print("⚠️ No disease context → fallback to generic")
    return "disease"

# =========================
# 🔹 NORMALIZE DISEASE
# =========================
def normalize_disease(d):
    return d.lower().replace("-", " ").strip()

# =========================
# 🔹 PUBMED SEARCH
# =========================
def search_pubmed(query, max_results=5):

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max_results
    }

    try:
        r = requests.get(url, params=params, timeout=20)
        data = r.json()

        count = int(data["esearchresult"]["count"])
        pmids = data["esearchresult"]["idlist"]

        return count, pmids

    except:
        return 0, []

# =========================
# 🔹 FETCH TITLES
# =========================
def fetch_titles(pmids):

    if not pmids:
        return []

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "json"
    }

    try:
        r = requests.get(url, params=params, timeout=20)
        data = r.json()

        titles = []
        for pmid in pmids:
            titles.append(data["result"][pmid]["title"])

        return titles

    except:
        return []

# =========================
# 🔹 RELEVANCE SCORING
# =========================
def relevance_score(titles):

    score = 0

    for t in titles:
        t_lower = t.lower()
        for kw in KEYWORDS:
            if kw in t_lower:
                score += 1

    return score / (len(titles) + 1e-6)

# =========================
# 🔹 LOG NORMALIZATION
# =========================
def log_norm(x):
    return np.log1p(x)

# =========================
# 🔹 CLASSIFICATION
# =========================
def classify(score):

    if score > 0.75:
        return "Strong"
    elif score > 0.4:
        return "Moderate"
    elif score > 0.1:
        return "Weak"
    else:
        return "Novel"

# =========================
# 🔹 MAIN FUNCTION
# =========================
def run_literature_validation():

    print("\n=== STEP 8: Literature Validation  ===")

    config = load_config()

    target = config.get("target_name", config["target_chembl_id"])
    target_name = config.get("target_name", "")
    user_disease = config.get("disease", None)

    DISEASE = normalize_disease(
        resolve_disease(user_disease, target_name)
    )

    print(f"🎯 Target: {target}")
    print(f"🧬 Disease context: {DISEASE}")

    df = pd.read_csv("drug_repurposing_full.csv")
    # 🔥 LIMIT TO TOP N (avoid huge PubMed load)
    TOP_N = 20
    df = df.sort_values("Score", ascending=False).head(TOP_N)

    print(f"📊 Using top {TOP_N} drugs for literature validation")

    results = []

    print("\n⏳ Mining PubMed...\n")

    for _, row in df.iterrows():

        drug = str(row["Name"])

        # =====================
        # 🔎 MULTI-QUERY
        # =====================
        q_dt = f"{drug} AND {target}"
        q_dd = f"{drug} AND {DISEASE}"
        q_td = f"{target} AND {DISEASE}"

        # =====================
        # 🔍 SEARCH
        # =====================
        h_dt, pmids_dt = search_pubmed(q_dt)
        h_dd, _ = search_pubmed(q_dd)
        h_td, _ = search_pubmed(q_td)

        # =====================
        # 📄 TITLES
        # =====================
        titles_dt = fetch_titles(pmids_dt)

        # =====================
        # 🧠 RELEVANCE
        # =====================
        rel = relevance_score(titles_dt)

        # =====================
        # 📊 COMPONENTS
        # =====================
        s_dt = log_norm(h_dt)
        s_dd = log_norm(h_dd)
        s_td = log_norm(h_td)

        # =====================
        # 🧮 LITERATURE SCORE
        # =====================
        raw_score = (
            ALPHA * s_dt +
            BETA  * s_dd +
            GAMMA * s_td +
            DELTA * rel
        )

        results.append({
            "Name": drug,
            "Hits_DT": h_dt,
            "Hits_DD": h_dd,
            "Hits_TD": h_td,
            "Relevance": rel,
            "Literature_raw": raw_score,
            "Top_PMIDs": ";".join(pmids_dt)
        })

        print(f"{drug:25} | Score={raw_score:.3f}")

        time.sleep(0.1)

    df_lit = pd.DataFrame(results)

    # =====================
    # 🔥 NORMALIZE (0–1)
    # =====================
    min_s = df_lit["Literature_raw"].min()
    max_s = df_lit["Literature_raw"].max()

    df_lit["Literature_Score"] = (
        (df_lit["Literature_raw"] - min_s) /
        (max_s - min_s + 1e-6)
    )

    # =====================
    # 🏷 CLASSIFICATION
    # =====================
    df_lit["Evidence"] = df_lit["Literature_Score"].apply(classify)

    # =====================
    # 🔗 MERGE
    # =====================
    df_final = df.merge(df_lit, on="Name", how="left")

    df_final.to_csv("literature_validation.csv", index=False)

    print("\n🎯 Step 8 completed successfully !")

    return df_final


# =========================
# ▶ RUN
# =========================
if __name__ == "__main__":
    run_literature_validation()
