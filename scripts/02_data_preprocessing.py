"""
AIRepurposer - Step 2: Data Preprocessing

This script cleans and prepares ChEMBL bioactivity data
for machine learning (QSAR modeling).

Operations:
✔ Handles unit harmonization (nM, µM, µg/mL)
✔ Converts µg/mL → µM using molecular weight
✔ Removes invalid data safely
✔ Aggregates duplicates
✔ Computes pIC50

Author: Ghada Mahjoub 
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

# =========================
# 🔹 UNIT CONVERSION
# =========================
def normalize_ic50(row):

    value = row['standard_value']
    unit = str(row['standard_units']).strip()
    smiles = row['canonical_smiles']

    # nM → µM
    if unit == 'nM':
        return value / 1000

    # µM (already correct)
    if unit in ['uM', 'µM']:
        return value

    # µg/mL → µM (CORRECT conversion)
    if unit in ['ug.mL-1', 'µg/mL']:

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        mw = Descriptors.MolWt(mol)

        if mw == 0:
            return None

        return (value * 1000) / mw

    # unsupported units → discard
    return None


# =========================
# 🔹 MAIN PREPROCESSING
# =========================
def preprocess_data(input_file):

    print("\n=== Data Preprocessing Module ===")

    # 🔹 Load data
    print("\n📥 Loading dataset...")
    df = pd.read_csv(input_file)
    print(f"Initial number of records: {df.shape[0]}")

    # 🔹 Keep relevant columns
    required_cols = [
        'molecule_chembl_id',
        'canonical_smiles',
        'standard_value',
        'standard_units'
    ]
    df = df[required_cols]

    # 🔹 Remove missing values
    df = df.dropna(subset=['canonical_smiles', 'standard_value'])
    print(f"After removing missing values: {df.shape[0]}")

    # 🔹 Convert IC50 to numeric
    df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
    df = df.dropna(subset=['standard_value'])

    # 🔹 Remove invalid values
    df = df[df['standard_value'] > 0]
    print(f"After removing invalid IC50 values: {df.shape[0]}")

    # =========================
    # 🔥 UNIT NORMALIZATION
    # =========================
    print("\n🔬 Converting IC50 to µM (including µg/mL)...")

    df['IC50_uM'] = df.apply(normalize_ic50, axis=1)

    before = len(df)
    df = df.dropna(subset=['IC50_uM'])
    after = len(df)

    print(f"✔ Valid after unit conversion: {after} (removed {before - after})")

    # =========================
    # 🔹 AGGREGATE DUPLICATES
    # =========================
    print("\n📊 Aggregating duplicate compounds...")

    df_clean = (
        df.groupby('molecule_chembl_id')
        .agg({
            'canonical_smiles': 'first',
            'IC50_uM': 'median'
        })
        .reset_index()
    )

    print(f"✔ Unique compounds: {df_clean.shape[0]}")

    # =========================
    # 🔹 pIC50 CONVERSION
    # =========================
    df_clean['pIC50'] = -np.log10(df_clean['IC50_uM'] * 1e-6)

    print("\n✅ pIC50 conversion completed")

    # =========================
    # 🔹 FINAL PREVIEW
    # =========================
    print("\n📊 Final dataset preview:")
    print(df_clean.head())

    return df_clean


# =========================
# 🔹 RUN
# =========================
def run_preprocessing():

    input_file = input("\nEnter path to input CSV file: ").strip()

    df_clean = preprocess_data(input_file)

    # 💾 Save
    print("\n" + "="*50)
    print("💾 SAVE CLEAN DATASET")
    print("="*50)

    while True:
        save = input("Do you want to save the cleaned dataset? (y/n): ").strip().lower()
        if save in ['y', 'n']:
            break
        else:
            print("⚠️ Please enter 'y' or 'n'.")

    if save == 'y':
        output_file = input("Enter output filename (e.g., chembl_clean.csv): ").strip()
        df_clean.to_csv(output_file, index=False)
        print(f"✅ Clean dataset saved as: {output_file}")
    else:
        print("ℹ️ Data not saved.")

    print("\n🎯 Step 2 completed successfully !")

    return df_clean


if __name__ == "__main__":
    run_preprocessing()
