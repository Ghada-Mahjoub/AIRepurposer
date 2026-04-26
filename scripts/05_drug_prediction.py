"""
AIRepurposer - Step 5 

✔ Feature alignment (selector)
✔ Robust novelty ("Not Known")
✔ Additive scoring (interpretable)
✔ Uncertainty estimation (ensemble variance)
✔ Confidence from uncertainty
✔ Diversity filtering
✔ PCA chemical space visualization
✔ Publication-ready outputs

Author: Ghada Mahjoub 
"""

import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit import DataStructs
from rdkit import RDLogger

from sklearn.decomposition import PCA

RDLogger.DisableLog('rdApp.*')


# =========================
# 🔬 FEATURE GENERATOR
# =========================
def build_feature_generator(fp_size,selector):

    print(f"🧬 Fingerprint size: {fp_size}")

    generator = GetMorganGenerator(radius=2, fpSize=fp_size)
    expected_features = selector.n_features_in_

    print(f"🔍 Selector expects {expected_features} features")
    print(f"🔍 Generated features: {6 + fp_size}")
    
    def featurize(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        desc = [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol)
        ]

        fp = np.array(generator.GetFingerprint(mol))
        return np.concatenate([desc, fp])

    return featurize


# =========================
# 🧬 DIVERSITY FILTER
# =========================
def diversity_filter(df, threshold=0.85):

    print("\n🧬 Applying diversity filtering...")

    fps = []
    for smi in df["SMILES"]:
        mol = Chem.MolFromSmiles(smi)
        fps.append(Chem.RDKFingerprint(mol) if mol else None)

    selected = []
    for i, fp in enumerate(fps):
        if fp is None:
            continue

        keep = True
        for j in selected:
            sim = DataStructs.FingerprintSimilarity(fp, fps[j])
            if sim > threshold:
                keep = False
                break

        if keep:
            selected.append(i)

    print(f"✔ Diversity reduced: {len(df)} → {len(selected)}")
    return df.iloc[selected]


# =========================
# 🚀 MAIN FUNCTION
# =========================
def predict_drugs(input_file):

    print("\n=== Drug Repurposing Module ===")

    # =====================
    # LOAD MODEL + SELECTOR
    # =====================
    model = joblib.load("best_model.pkl")
    selector = joblib.load("feature_selector.pkl")

    # =====================
    # LOAD CONFIG
    # =====================
    with open("project_config.json") as f:
        config = json.load(f)
        
    # =====================
    # LOAD FEATURE CONFIG
    # =====================
    with open("feature_config.json") as f:
             feature_config = json.load(f)

    fp_size = feature_config["fp_size"]
    print(f"🧬 Loaded fp_size from training: {fp_size}")
    target_id = config["target_chembl_id"]
    known_set = set(str(x).strip() for x in config.get("known_active_compounds", []))

    print(f"🎯 Target: {target_id}")
    print(f"🧠 Known compounds: {len(known_set)}")

    # =====================
    # LOAD DATA
    # =====================
    df = pd.read_csv(input_file)

    print(f"\n📊 Initial drugs: {len(df)}")

    df = df.dropna(subset=["SMILES"]).copy()
    df["Mol"] = df["SMILES"].apply(Chem.MolFromSmiles)
    df = df[df["Mol"].notna()].copy()

    print(f"✔ After cleaning: {len(df)} drugs")

    if "Name" not in df.columns:
        df["Name"] = "Unknown"

    chembl_col = "ChEMBL ID" if "ChEMBL ID" in df.columns else None

    # =====================
    # FEATURES
    # =====================
    print("\n⏳ Generating features...")

    featurize = build_feature_generator(fp_size,selector)

    features = []
    valid_idx = []

    for i, smi in enumerate(df["SMILES"]):
        feat = featurize(smi)
        if feat is not None:
            features.append(feat)
            valid_idx.append(i)

    X = np.array(features)
    df = df.iloc[valid_idx].reset_index(drop=True)

    # 🔥 Apply selector
    X = selector.transform(X)
    print(f"✔ Generated feature matrix shape: {X.shape}")
    print(f"✔ Valid drugs: {len(df)}")

    # =====================
    # 🔥 ENSEMBLE PREDICTION + UNCERTAINTY
    # =====================
    preds = np.array([est.predict(X) for est in model.estimators_])
    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)

    df["Predicted_pIC50"] = mean_pred
    df["Uncertainty"] = std_pred

    # =====================
    # CONFIDENCE (UNCERTAINTY-BASED)
    # =====================
    def confidence(u):
        if u < 0.3: return "High"
        elif u < 0.6: return "Moderate"
        else: return "Low"

    df["Confidence"] = df["Uncertainty"].apply(confidence)

    # =====================
    # NOVELTY (FIXED)
    # =====================
    def novelty(x):
        if pd.isna(x): return "Unknown"
        return "Known" if str(x).strip() in known_set else "Not Known"

    if chembl_col:
        df["Novelty"] = df[chembl_col].apply(novelty)
    else:
        df["Novelty"] = "Unknown"

    # =====================
    # SCORE (ADD + BONUS)
    # =====================
    def compute_score(row):
        bonus = 0.5 if row["Novelty"] == "Not Known" else 0
        return row["Predicted_pIC50"] + bonus

    df["Score"] = df.apply(compute_score, axis=1)


        # =====================
    # SORT
    # =====================
    df = df.sort_values(by="Score", ascending=False)

    # =====================
    # DIVERSITY FILTER
    # =====================
    top = df.head(200)
    diverse = diversity_filter(top)

    # =====================
    # SAVE FULL RESULTS
    # =====================
    df.to_csv("drug_repurposing_full.csv", index=False)
    diverse.to_csv("top_diverse_candidates.csv", index=False)

    docking = diverse[diverse["Predicted_pIC50"] >= 7]
    docking.to_csv("docking_candidates.csv", index=False)

    # =====================
    # DISPLAY
    # =====================
    print("\n🏆 Top candidates:")
    print(
        diverse[
            ["Name", "Predicted_pIC50", "Uncertainty", "Confidence", "Novelty", "Score"]
        ].head(10)
    )

    # =====================
    # 📄 SAVE TOP TABLE (PAPER READY)
    # =====================
    top_table = diverse[
        ["Name", "Predicted_pIC50", "Uncertainty", "Confidence", "Novelty", "Score"]
    ].head(20)

    # round values for clean presentation
    top_table = top_table.round({
        "Predicted_pIC50": 2,
        "Uncertainty": 2,
        "Score": 2
    })

    top_table.to_csv("top20_candidates_table.csv", index=False)

    print("📄 Top candidates table saved: top20_candidates_table.csv")

    # =====================
    # 📊 TOP CANDIDATES BARPLOT
    # =====================
    top10 = diverse.head(10)

    plt.figure(figsize=(10, 6))
    plt.yticks(fontsize=9)

    colors = plt.cm.plasma(top10["Score"] / top10["Score"].max())

    plt.barh(
        top10["Name"],
        top10["Score"],
        color=colors,
        edgecolor="black"
    )

    plt.gca().invert_yaxis()

    # annotate values
    for i, v in enumerate(top10["Score"]):
        plt.text(v + 0.05, i, f"{v:.2f}", va="center")

    plt.xlabel("Repurposing Score")
    plt.title("Top Drug Repurposing Candidates")

    plt.tight_layout()
    plt.savefig("top_candidates.png", dpi=600)
    plt.show()
    # =====================
    # 📊 PCA CHEMICAL SPACE
    # =====================
    print("\n📊 Generating chemical space plot...")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(6,5))
    sc = plt.scatter(X_pca[:,0], X_pca[:,1],
                     c=df["Predicted_pIC50"],
                     cmap="viridis",
                     alpha=0.7)

    plt.colorbar(sc, label="Predicted pIC50")
    plt.title("Chemical Space of DrugBank Compounds")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.tight_layout()
    plt.savefig("chemical_space.png", dpi=600)
    plt.show()

    print("\n🎯 Step 5 completed successfully!")

    return diverse


# ▶ RUN
if __name__ == "__main__":
    file = input("Enter DrugBank dataset file: ").strip()
    predict_drugs(file)
