"""
AIRepurposer - Step 3 (IMPROVED)

✔ Descriptor calculation
✔ Salt removal
✔ Adaptive fingerprints
✔ RDKit error handling

Author: Ghada Mahjoub
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit import RDLogger

# 🔇 Silence RDKit warnings
RDLogger.DisableLog('rdApp.*')


# =========================
# 🔹 Clean molecule
# =========================
def get_largest_fragment(smiles):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fragments = Chem.GetMolFrags(mol, asMols=True)

    if len(fragments) == 0:
        return None

    return max(fragments, key=lambda m: m.GetNumAtoms())


# =========================
# 🔹 Descriptor calculation
# =========================
def compute_descriptors(mol):

    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol)
    ]


# =========================
# 🔹 MAIN FUNCTION
# =========================
def feature_engineering(input_file):

    print("\n=== Feature Engineering Module (Improved) ===")

    df = pd.read_csv(input_file)
    print(f"Loaded dataset: {df.shape[0]} compounds")

    # 🔬 Adaptive fingerprint size
    n_samples = len(df)
    fp_size = min(1024, max(256, n_samples))

    print(f"🧬 Fingerprint size: {fp_size}")

    generator = GetMorganGenerator(radius=2, fpSize=fp_size)

    features = []
    valid_indices = []

    print("\n⏳ Processing molecules...")

    for i, row in df.iterrows():

        smiles = row["canonical_smiles"]

        mol = get_largest_fragment(smiles)

        if mol is None:
            continue

        try:
            desc = compute_descriptors(mol)
            fp = np.array(generator.GetFingerprint(mol))

            combined = np.concatenate([desc, fp])

            features.append(combined)
            valid_indices.append(i)

        except:
            continue

    df_valid = df.loc[valid_indices].reset_index(drop=True)

    feature_names = (
        ["MolWt", "LogP", "NumHDonors", "NumHAcceptors", "TPSA", "NumRotatableBonds"] +
        [f"FP_{i}" for i in range(fp_size)]
    )

    df_features = pd.DataFrame(features, columns=feature_names)

    df_final = pd.concat([df_valid, df_features], axis=1)

    print(f"\n✅ Valid molecules: {df_final.shape[0]}")
    print(f"❌ Removed: {df.shape[0] - df_final.shape[0]}")

    print("\n📊 Feature matrix shape:", df_features.shape)

    return df_final


# =========================
# ▶ RUN
# =========================
def run_feature_engineering():

    input_file = input("\nEnter cleaned dataset file: ").strip()

    df_final = feature_engineering(input_file)

    save = input("\nSave dataset? (y/n): ").lower()

    if save == "y":
        output = input("Filename (e.g.,features.csv): ").strip()
        df_final.to_csv(output, index=False)
        print("✅ Saved")

    print("\n🎯 Step 3 completed successfully !")


if __name__ == "__main__":
    run_feature_engineering()
