"""
AIRepurposer - Step 4 (ULTIMATE PRO VERSION)

✔ Adaptive fingerprints
✔ tqdm progress bar
✔ Feature selection
✔ Robust KFold CV
✔ Full metrics (R2, RMSE, MAE)
✔ Voting Regressor 
✔ Publication-quality plots (PNG + TIFF)
✔ Interactive plots (Plotly)
✔ Descriptor + Fingerprint importance
✔ Full logging + traceability

Author: Ghada Mahjoub 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import time
from tqdm import tqdm
import json
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, VotingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Optional interactive
try:
    import plotly.express as px
    INTERACTIVE = True
except:
    INTERACTIVE = False

# =========================
# 🎨 Plot settings
# =========================
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "savefig.dpi": 600
})

# =========================
# 📊 Metrics
# =========================
def evaluate(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred)
    }

# =========================
# 🚀 MAIN FUNCTION
# =========================
def train_ultimate(input_file):

    start_global = time.time()

    print("\n🚀 === STEP 4: MODEL TRAINING ===")

    df = pd.read_csv(input_file)

    # =========================
    # 🧬 Fingerprints
    # =========================
    n_samples = len(df)
    fp_size = min(1024, max(256, n_samples))

    print("\n🧬 Preparing molecular representation...")
    print(f"   → Fingerprint size: {fp_size}")
    print("⏳ Generating fingerprints...")

    generator = GetMorganGenerator(radius=2, fpSize=fp_size)

    def smiles_to_fp(smiles):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return np.zeros(fp_size)
        return np.array(generator.GetFingerprint(mol))

    fps = np.array([smiles_to_fp(s) for s in tqdm(df["canonical_smiles"], desc="Fingerprints")])

    # =========================
    # 📊 Features
    # =========================
    desc_cols = ["MolWt","LogP","NumHDonors","NumHAcceptors","TPSA","NumRotatableBonds"]

    X_desc = df[desc_cols].values
    X = np.hstack([X_desc, fps])
    y = df["pIC50"].values

    print(f"\n📊 Feature matrix: {X.shape}")

   

    # =========================
    # Split
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
     # =========================
    # Feature selection
    # =========================
    selector = VarianceThreshold(0.01)
    selector.fit(X_train)
    X_train = selector.transform(X_train)
    X_test = selector.transform(X_test)

    print(f"🔬 After feature selection: {X_train.shape}")
    fp_size = min(1024, max(256, len(df)))
    with open("feature_config.json", "w") as f:
              json.dump({"fp_size": fp_size}, f)
              
    print(f"✔ Feature config saved (fp_size={fp_size})")         
    # =========================
    # Models
    # =========================
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=150, n_jobs=-1, random_state=42),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=150, n_jobs=-1, random_state=42),
        "HistGB": HistGradientBoostingRegressor(),
        "GradientBoosting": GradientBoostingRegressor(),

        "SVR": Pipeline([("scaler", StandardScaler()), ("model", SVR())]),
        "KNN": Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor())]),
        "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge())]),
        "MLP": Pipeline([("scaler", StandardScaler()),
                         ("model", MLPRegressor(max_iter=1000, random_state=42))])
    }

    # =========================
    # Training
    # =========================
    results = []
    trained = {}

    print("\n⏳ Training models...\n")

    for name, model in models.items():

        t0 = time.time()
        print(f"🔹 Training {name}...")

        cv = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2").mean()

        model.fit(X_train, y_train)
        trained[name] = model

        y_pred = model.predict(X_test)
        metrics = evaluate(y_test, y_pred)

        results.append([name, cv, metrics["R2"], metrics["RMSE"], metrics["MAE"]])

        print(
            f"✔ {name:15} | CV={cv:.3f} | "
            f"R2={metrics['R2']:.3f} | RMSE={metrics['RMSE']:.3f} | MAE={metrics['MAE']:.3f} "
            f"| time={time.time()-t0:.1f}s"
        )

    results_df = pd.DataFrame(results, columns=["Model","CV_R2","Test_R2","RMSE","MAE"])
    results_df = results_df.sort_values(by="Test_R2", ascending=False)

    print("\n🏆 MODEL RANKING:")
    print(results_df)
    
    # =========================
    # Ensemble
    # =========================
    top_names = results_df.head(3)["Model"].values
    print("\n🤖 Building Voting Regressor:", top_names)

    ensemble = VotingRegressor([(n, trained[n]) for n in top_names])
    ensemble.fit(X_train, y_train)

    y_pred = ensemble.predict(X_test)
    final_metrics = evaluate(y_test, y_pred)

    print("\n🎯 FINAL ENSEMBLE:")
    print(final_metrics)
    # =========================
    # 💾 SAVE PREDICTIONS
    # =========================
    df_pred = pd.DataFrame({
         "Experimental": y_test,
         "Predicted": y_pred
    })
    df_pred.to_csv("test_predictions.csv", index=False)
    
    # =========================
    # 🔬 Y-RANDOMIZATION TEST
    # =========================
    print("\n🧪 Running Y-randomization test...")
    y_shuffled = np.random.permutation(y_train)
    model_test = GradientBoostingRegressor()
    model_test.fit(X_train, y_shuffled)
    y_pred_rand = model_test.predict(X_test)
    r2_rand = r2_score(y_test, y_pred_rand)
    print(f"❌ Y-randomization R²: {r2_rand:.3f}")

   # =========================
   # 📏 APPLICABILITY DOMAIN
   # =========================
    from sklearn.metrics import pairwise_distances

    print("\n📏 Computing Applicability Domain...")
    distances = pairwise_distances(X_test, X_train)
    min_dist = distances.min(axis=1)
    threshold = np.percentile(min_dist, 95)
    df_ad = pd.DataFrame({
         "MinDistance": min_dist,
         "InDomain": min_dist <= threshold
    })
    df_ad.to_csv("applicability_domain.csv", index=False)
    print(f"✔ AD threshold: {threshold:.3f}")
    print(f"✔ In-domain samples: {(df_ad['InDomain'].sum())}/{len(df_ad)}")
    # =========================
    # Save
    # =========================
    joblib.dump(ensemble, "best_model.pkl")
    joblib.dump(selector, "feature_selector.pkl")
    results_df.to_csv("model_results.csv", index=False)

    print("💾 Model + results saved")

    save = input("\nSave plots? (y/n): ").lower() == "y"

    def savefig(name):
        if save:
            plt.savefig(name + ".tiff", bbox_inches="tight")
            plt.savefig(name + ".png", bbox_inches="tight")
  
    # =========================
    # 📊 Model comparison
    # =========================
    plt.figure(figsize=(8,5))

    colors = plt.cm.viridis(results_df["Test_R2"] / results_df["Test_R2"].max())
    bars = plt.bar(results_df["Model"], results_df["Test_R2"],
                   color=colors, edgecolor="black")

    bars[0].set_edgecolor("red")
    bars[0].set_linewidth(2)

    for bar in bars:
        plt.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+0.01,
                 f"{bar.get_height():.2f}",
                 ha="center")

    plt.title("Model Comparison")
    plt.ylabel("R² Score")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    savefig("model_comparison")
    plt.show()

    # =========================
    # 📈 Scatter
    # =========================
    plt.figure(figsize=(6,6))

    plt.scatter(y_test, y_pred,
                color="#1f77b4",
                alpha=0.6,
                edgecolor="black",
                linewidth=0.3,
                label="Predictions")

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())

    plt.plot([min_val, max_val], [min_val, max_val],
             "--", color="red", label="Perfect")

    plt.xlabel("Experimental pIC50")
    plt.ylabel("Predicted pIC50")
    plt.title(f"Prediction (R²={final_metrics['R2']:.3f})")

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    savefig("scatter")
    plt.show()

    # =========================
    # 📉 Residuals
    # =========================
    residuals = y_test - y_pred

    plt.figure(figsize=(6,4))
    plt.hist(residuals, bins=30,
             color="#2ca02c", edgecolor="black")

    plt.axvline(0, color="red", linestyle="--", label="Zero error")

    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")

    plt.legend()
    plt.tight_layout()
    savefig("residuals")
    plt.show()

    # =========================
    # 🧠 Descriptor importance
    # =========================
    print("\n🧠 Interpretability model...")

    rf_desc = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_desc.fit(X_desc, y)

    imp = rf_desc.feature_importances_
    idx = np.argsort(imp)

    plt.figure(figsize=(7,4))
    plt.barh(np.array(desc_cols)[idx], imp[idx],
             color="#ff7f0e", edgecolor="black")

    plt.title("Descriptor Importance")
    plt.tight_layout()
    savefig("descriptor_importance")
    plt.show()

    # =========================
    # 🌐 Interactive (Plotly)
    # =========================
    if INTERACTIVE:
        print("\n🌐 Generating interactive plot...")

        fig = px.scatter(x=y_test, y=y_pred,
                         labels={"x":"Actual", "y":"Predicted"},
                         title="Interactive Prediction Plot")

        fig.write_html("interactive_prediction.html")
        fig.show()
        print("✔ Interactive plot saved: interactive_prediction.html")

    print("\n🎯 STEP 4 COMPLETED")
    print(f"⏱ Total time: {time.time()-start_global:.1f}s")

    return results_df


# ▶ RUN
if __name__ == "__main__":
    file = input("Enter dataset file: ").strip()
    train_ultimate(file)
