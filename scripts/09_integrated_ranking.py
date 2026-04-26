"""
AIRepurposer — Step 9 FINAL (CONSENSUS RANKING )

✔ Integrates ML + Docking + Network + Literature
✔ Uses normalized continuous scores
✔ Robust missing-data handling
✔ Weighted consensus scoring
✔ Confidence estimation
✔ Publication-ready outputs

Author: Ghada Mahjoub 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 🔹 WEIGHTS (JUSTIFIED)
# =========================
W_ML   = 0.35
W_DOCK = 0.30
W_NET  = 0.25
W_LIT  = 0.10

# =========================
# 🔹 NORMALIZATION
# =========================
def normalize(series):

    series = series.fillna(series.median())

    if series.max() == series.min():
        return pd.Series([0.5]*len(series))

    return (series - series.min()) / (series.max() - series.min())

# =========================
# 🔹 LOAD DATA
# =========================
def load_data():

    df_pred = pd.read_csv("drug_repurposing_full.csv")
    df_dock = pd.read_csv("final_docking_results.csv")
    df_net  = pd.read_csv("drug_network_scores.csv")
    df_lit  = pd.read_csv("literature_validation.csv")

    return df_pred, df_dock, df_net, df_lit

# =========================
# 🔹 MERGE DATA
# =========================
def merge_data(df_pred, df_dock, df_net, df_lit):

    df = df_pred.copy()

    # =====================
    # 🔥 STANDARDIZE COLUMN NAMES
    # =====================
    if "Drug" in df_net.columns:
        df_net = df_net.rename(columns={"Drug": "Name"})

    # literature already has Name → no change needed

    # =====================
    # 🔗 MERGE STEP BY STEP
    # =====================
    df = df.merge(
        df_dock[["Name", "Docking_Affinity"]],
        on="Name", how="left"
    )

    df = df.merge(
        df_net[["Name", "Network_Score"]],
        on="Name", how="left"
    )

    df = df.merge(
        df_lit[["Name", "Literature_Score"]],
        on="Name", how="left"
    )

    return df

# =========================
# 🔹 COMPUTE SCORES
# =========================
def compute_scores(df):

    # =====================
    # ML
    # =====================
    df["ML_score"] = normalize(df["Predicted_pIC50"])

    # =====================
    # Docking (invert)
    # =====================
    df["Docking_score"] = normalize(-df["Docking_Affinity"].fillna(df["Docking_Affinity"].median()))

    # =====================
    # Network
    # =====================
    if  "Network_Score_norm" in df.columns:
         df["Network_score"] = df["Network_Score_norm"]
    else:
         df["Network_score"] = normalize(df["Network_Score"])
    

    # =====================
    # Literature (already normalized)
    # =====================
    df["Literature_score"] = normalize(df["Literature_Score"].fillna(0))

    # =====================
    # FINAL CONSENSUS SCORE
    # =====================
    df["Novelty_penalty"] = 1 - df["Literature_score"]
    df["Final_score"] = (
        W_ML   * df["ML_score"] +
        W_DOCK * df["Docking_score"] +
        W_NET  * df["Network_score"] +
        W_LIT  * df["Literature_score"] * 0.5 +
        0.05   * df["Novelty_penalty"]
    )

    # =====================
    # 🔥 CONFIDENCE SCORE (NEW)
    # =====================
    df["Score_std"] = df[
        ["ML_score", "Docking_score", "Network_score", "Literature_score"]
    ].std(axis=1)

    df["Confidence"] = 1 - df["Score_std"]

    return df

# =========================
# 🔹 PLOT FINAL RESULTS
# =========================
def plot_final(df):

    top = df.sort_values("Final_score", ascending=False).head(10)

    plt.figure(figsize=(10,6))

    colors = plt.cm.viridis(top["Final_score"])

    bars = plt.barh(
        top["Name"],
        top["Final_score"],
        color=colors,
        edgecolor="black"
    )

    plt.xlabel("Final Integrated Score")
    plt.title("Top Drug Candidates — Multi-Layer Consensus Ranking")
    plt.grid(axis='x', linestyle='--', alpha=0.3)

    plt.gca().invert_yaxis()

    for i, v in enumerate(top["Final_score"]):
        plt.text(v + 0.01, i, f"{v:.2f}", va="center")

    plt.tight_layout()
    plt.savefig("final_ranking.png", dpi=600)
    plt.savefig("final_ranking.svg")
    plt.show()

# =========================
# 🔹 MAIN
# =========================
def main():

    print("\n=== STEP 9: FINAL CONSENSUS RANKING ===")

    df_pred, df_dock, df_net, df_lit = load_data()

    print("📊 Merging datasets...")
    df = merge_data(df_pred, df_dock, df_net, df_lit)

    print("📊 Computing scores...")
    df = compute_scores(df)

    df = df.sort_values(by="Final_score", ascending=False)

    # =====================
    # SAVE
    # =====================
    df.to_csv("final_ranked_drugs.csv", index=False)

    print("\n🏆 TOP DRUGS:")
    print(df[[
        "Name",
        "Final_score",
        "Confidence",
        "ML_score",
        "Docking_score",
        "Network_score",
        "Literature_score"
    ]].head(10))

    print("\n📊 Generating final figure...")
    plot_final(df)

    print("\n🎯 Step 9 COMPLETE !")


# =========================
# ▶ RUN
# =========================
if __name__ == "__main__":
    main()
