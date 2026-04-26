# ==========================================
# PLOTS FOR DOCKING (FINAL - PUBLICATION)
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# ==========================================
# LOAD + CLEAN DATA
# ==========================================
def load_data():

    df = pd.read_csv("final_docking_results.csv")

    # Remove invalid docking values
    df = df[df["Docking_Affinity"] < 0]
    df = df[df["Docking_Affinity"] > -15]

    print(f"\n✅ Clean dataset: {len(df)} compounds")

    return df


# ==========================================
# CORRELATION PLOT
# ==========================================
def plot_correlation(df):

    x = df["Predicted_pIC50"]
    y = -df["Docking_Affinity"]

    plt.figure(figsize=(7,6))

    scatter = plt.scatter(
            x, y,
            c=df["Score"],
            cmap="inferno",
            s=90,
            edgecolor="black",
            alpha=0.85
    )
    slope, intercept, r, p, _ = linregress(x, y)
    plt.text(
         0.05, 0.95,
         f"R = {r:.2f}",
         transform=plt.gca().transAxes,
         fontsize=11,
         verticalalignment='top'
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label("Repurposing Score", fontsize=11)

    # 🔥 Quadrant lines
    plt.axvline(7, linestyle="--", color="gray", alpha=0.5)
    plt.axhline(6.5, linestyle="--", color="gray", alpha=0.5)

    # 🔥 Highlight top 5
    top5 = df.sort_values("Score", ascending=False).head(5)

    plt.scatter(
        top5["Predicted_pIC50"],
        -top5["Docking_Affinity"],
        color="red",
        s=120,
        edgecolor="black",
        label="Top candidates"
    )

    for _, row in top5.iterrows():
        plt.text(
            row["Predicted_pIC50"],
            -row["Docking_Affinity"],
            row["Name"],
            fontsize=8,
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
        )

    plt.xlabel("Predicted Activity (pIC50)", fontsize=12)
    plt.ylabel("Binding Strength (Higher = Stronger Binding)", fontsize=12)

    plt.title("ML Predictions vs Docking: Candidate Landscape", fontsize=13)

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig("correlation_plot.png", dpi=600, bbox_inches="tight")
    plt.show()


# ==========================================
# SCORE DISTRIBUTION
# ==========================================
def plot_score_distribution(df):

    plt.figure(figsize=(6,5))

    plt.hist(
        df["Score"],
        bins=20,
        color="#8E44AD",  # purple
        edgecolor="black",
        alpha=0.85
    )

    mean_val = df["Score"].mean()

    plt.axvline(
        mean_val,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean_val:.2f}"
    )

    plt.xlabel("Repurposing Score\n(Combined ML ranking metric)", fontsize=11)
    plt.ylabel("Number of Compounds", fontsize=11)

    plt.title("Distribution of Repurposing Scores", fontsize=13)

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("score_distribution.png", dpi=600, bbox_inches="tight")
    plt.show()


# ==========================================
# DOCKING DISTRIBUTION
# ==========================================
def plot_docking_distribution(df):

    plt.figure(figsize=(6,5))

    plt.hist(
        df["Docking_Affinity"],
        bins=20,
        color="#E67E22",  # orange
        edgecolor="black",
        alpha=0.85
    )

    plt.axvline(
        -7,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Strong binding threshold (-7 kcal/mol)"
    )
    #plt.text(-7.5, 1,"Small sample (n=10)",fontsize=9, alpha=0.7)
    
    plt.xlabel("Docking Affinity (kcal/mol)\n(Lower = stronger binding)", fontsize=11)
    plt.ylabel("Number of Compounds", fontsize=11)

    plt.title("Distribution of Docking Affinities", fontsize=13)

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("docking_distribution.png", dpi=600, bbox_inches="tight")
    plt.show()


# ==========================================
# TOP DRUG SELECTION
# ==========================================
def select_top_compounds(df, n=5):

    # =========================
    # Clean data
    # =========================
    df = df.copy()

    # Keep only valid docking values
    df = df[df["Docking_Affinity"].notna()]
    df = df[df["Docking_Affinity"] < 0]

    # Remove duplicates (same drug name)
    df = df.drop_duplicates(subset="Name")

    # =========================
    # Multi-criteria filtering
    # =========================
    top = df[
        (df["Docking_Affinity"] <= -7)
    ].sort_values(
        by=["Score", "Docking_Affinity"],
        ascending=[False, True]   # high score + strong binding
    )

    topN = top.head(n)

    # =========================
    # Keep clean columns (paper-ready)
    # =========================
    clean_cols = [
        "Name",
        "DrugBank ID",
        "Predicted_pIC50",
        "Docking_Affinity",
        "Score",
        "Confidence",
        "Novelty"
    ]

    clean_cols = [c for c in clean_cols if c in topN.columns]

    topN_clean = topN[clean_cols]

    print("\n🔥 TOP CANDIDATES:")
    print(topN_clean)

    # Save clean + full versions
    topN_clean.to_csv("top_candidates_clean.csv", index=False)
    topN.to_csv("top_candidates_full.csv", index=False)

    return topN
# ==========================================
def plot_top5(df, top5):

    plt.figure(figsize=(6,5))

    plt.scatter(
        df["Predicted_pIC50"],
        -df["Docking_Affinity"],
        color="lightgray",
        alpha=0.5,
        label="All compounds"
    )

    plt.scatter(
        top5["Predicted_pIC50"],
        -top5["Docking_Affinity"],
        color="red",
        s=100,
        edgecolor="black",
        label="Top 5 candidates"
    )

    for _, row in top5.iterrows():
        plt.text(
            row["Predicted_pIC50"],
            -row["Docking_Affinity"],
            f"{row['Name']} ({row['Score']:.2f})",
            fontsize=8
        )
    plt.xlabel("Predicted pIC50")
    plt.ylabel("Binding Strength (Higher = Stronger Binding)")
    plt.title("Top Candidate Compounds")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("top5_highlight.png", dpi=600, bbox_inches="tight")
    plt.show()

# ==========================================
# MAIN
# ==========================================
def main():

    df = load_data()

    # 🔥 SELECT TOP COMPOUNDS FIRST
    top5 = select_top_compounds(df)

    # Plots (use full dataset)
    plot_correlation(df)
    plot_score_distribution(df)
    plot_docking_distribution(df)
    plot_top5(df, top5)

if __name__ == "__main__":
    main()
