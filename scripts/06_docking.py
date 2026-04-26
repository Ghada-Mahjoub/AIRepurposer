# -*- coding: utf-8 -*-

import os
import json
import time
import sys
import requests
import pandas as pd
import numpy as np
import shutil 
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

# =========================
# SETUP
# =========================
def reset_workspace():
    print("\n🧹 Cleaning previous docking workspace...")

    for folder in ["protein", "ligands", "results"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)  # 🔥 DELETE COMPLETELY

        os.makedirs(folder)  # 🔥 RECREATE EMPTY

    print("✅ Workspace ready (fresh run)\n")

# =========================
# SAFE REQUEST
# =========================
def safe_get_json(url, retries=5):
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                return r.json()
        except:
            pass
        time.sleep(2)
    raise RuntimeError(f"API failed: {url}")

# =========================
# LOAD TARGET
# =========================
def load_target():
    with open("project_config.json") as f:
        cfg = json.load(f)

    chembl_id = cfg["target_chembl_id"]
    print(f"\n🎯 Target: {chembl_id}")
    return chembl_id

# =========================
# GET UNIPROT
# =========================
def get_uniprot(chembl_id):
    url = f"https://www.ebi.ac.uk/chembl/api/data/target/{chembl_id}.json"
    data = safe_get_json(url)

    accession = data["target_components"][0]["accession"]
    print(f"🔗 UniProt: {accession}")
    return accession

# =========================
# GET PDB IDS
# =========================
def get_pdb_ids(uniprot):

    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                "operator": "exact_match",
                "value": uniprot
            }
        },
        "return_type": "entry"
    }

    res = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=query)
    data = res.json()

    pdb_ids = [r["identifier"] for r in data.get("result_set", [])]

    if not pdb_ids:
        raise ValueError("No PDB found")

    print(f"🧬 PDB candidates: {len(pdb_ids)}")
    return pdb_ids

# =========================
# GET PDB INFO
# =========================
def get_pdb_info(pdb_id):

    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    data = safe_get_json(url)

    resolution = data.get("rcsb_entry_info", {}).get("resolution_combined", [99])[0]
    method = data.get("exptl", [{}])[0].get("method", "")

    return {"pdb": pdb_id, "resolution": resolution, "method": method}

# =========================
# SELECT BEST PDB
# =========================
def select_pdb(pdb_ids):

    infos = [get_pdb_info(p) for p in pdb_ids]
    df = pd.DataFrame(infos)

    xray = df[df["method"].str.contains("X-ray", case=False, na=False)]

    if not xray.empty:
        df = xray
    else:
        print("⚠ No X-ray structure → using best available")

    df = df.sort_values("resolution")

    best = df.iloc[0]

    print(f"✅ Selected PDB: {best['pdb']} ({best['resolution']} Å)")
    return best["pdb"]

# =========================
# DOWNLOAD PDB
# =========================
def download_pdb(pdb_id):

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    r = requests.get(url)

    if r.status_code != 200:
        raise RuntimeError("Download failed")

    with open("protein/protein.pdb", "wb") as f:
        f.write(r.content)


# =========================
# SELECT DRUGS
# =========================
def select_drugs(mode="interactive"):

    df = pd.read_csv("drug_repurposing_full.csv")

    print(f"\n📊 Total candidates: {len(df)}")

    # =========================
    # 🔥 AUTO MODE (DEFAULT)
    # =========================
    if mode == "auto":
        selected = df.sort_values("Score", ascending=False).head(50)
        print("\n⚙ AUTO MODE → Top 50 by Repurposing Score")
        print("\n🏆 Selected compounds summary:")
        print(selected[["Name", "Predicted_pIC50", "Score"]].head(10))
        return selected

    # =========================
    # 🔹 INTERACTIVE MODE
    # =========================
    print("\nSelect docking strategy:")
    print("1 → Top 50 by Score")
    print("2 → Activity threshold (Predicted_pIC50 ≥ 7)")
    print("3 → High confidence + Novel")
    print("4 → Custom number")

    c = input("Choice: ").strip()

    if c == "1":
        selected = df.sort_values("Score", ascending=False).head(50)

    elif c == "2":
        subset = df[df["Predicted_pIC50"] >= 7]
        print(f"✔ Found {len(subset)} compounds with pIC50 ≥ 7")
        selected = subset.sort_values("Score", ascending=False).head(50)

    elif c == "3":
        subset = df[
            (df["Confidence"] == "High") &
            (df["Novelty"] == "Not Known")
        ]
        print(f"✔ Found {len(subset)} high-confidence novel compounds")
        selected = subset.sort_values("Score", ascending=False).head(20)

    elif c == "4":
        while True:
            try:
                n = int(input("Enter number of compounds to dock: "))
                if n > 0:
                    break
            except:
                print("⚠️ Enter a valid number")

        selected = df.sort_values("Score", ascending=False).head(n)

    else:
        raise ValueError("❌ Invalid choice")

    print("\n🏆 Selected compounds summary:")
    print(selected[["Name", "Predicted_pIC50", "Score"]].head(10))

    return selected

# =========================
# CLEAN PROTEIN
# =========================
def clean_protein():

    lines = open("protein/protein.pdb").readlines()
    clean = [l for l in lines if l.startswith("ATOM") and l[16] in [" ", "A"]]

    with open("protein/protein_clean.pdb", "w") as f:
        f.writelines(clean)

# =========================
# PREPARE PROTEIN
# =========================
def prepare_protein():

    cmd = (
        "obabel protein/protein_clean.pdb "
        "-O protein/protein.pdbqt "
        "-xr -p 7.4 --partialcharge gasteiger"
    )
    os.system(cmd)

# =========================
# POCKET DETECTION
# =========================
def get_best_pocket():

    # =========================
    # Clean old fpocket outputs
    # =========================
    for d in os.listdir("protein"):
        if d.endswith("_out"):
            os.system(f"rm -rf protein/{d}")

    print("\n🔍 Running fpocket...")
    os.system("fpocket -f protein/protein_clean.pdb")

    # =========================
    # Find fpocket output folder
    # =========================
    folders = [
        os.path.join("protein", d)
        for d in os.listdir("protein")
        if d.endswith("_out")
    ]

    if not folders:
        raise RuntimeError("❌ fpocket failed: no output folder found")

    folder = folders[0]

    print(f"📂 Found fpocket folder: {folder}")

    # =========================
    # Find info file dynamically
    # =========================
    info_files = [
        f for f in os.listdir(folder)
        if f.endswith("_info.txt")
    ]

    if not info_files:
        raise RuntimeError(f"❌ No info file found in {folder}")

    info_path = os.path.join(folder, info_files[0])

    print(f"📁 Using info file: {info_path}")

    # =========================
    # Parse pocket scores
    # =========================
    best_id, best_score = None, -1

    with open(info_path) as f:
        for line in f:

            if "Pocket" in line:
                pid = int(line.split()[1])

            elif "Druggability Score" in line:
                score = float(line.split(":")[1])

                if score > best_score:
                    best_score = score
                    best_id = pid

    if best_id is None:
        raise RuntimeError("❌ No valid pocket found")

    # =========================
    # Build pocket path
    # =========================
    pocket_path = os.path.join(
        folder,
        "pockets",
        f"pocket{best_id}_vert.pqr"
    )

    print(f"✅ Selected pocket: pocket{best_id} (score={best_score:.3f})")

    return pocket_path

# =========================
# CENTER
# =========================
def get_center(pocket):

    coords = [
        [float(l[30:38]), float(l[38:46]), float(l[46:54])]
        for l in open(pocket) if l.startswith("ATOM")
    ]

    center = np.mean(coords, axis=0)
    print(f"📍 Docking center: {center}")
    return center

# =========================
# PREPARE LIGANDS
# =========================
def prepare_ligands(df):

    print("\n🧪 Preparing ligands...")

    for _, row in df.iterrows():

        name = row["Name"]
        safe_name = name.replace(" ", "_")
        mol = Chem.MolFromSmiles(row["SMILES"])

        if mol is None:
            print(f"⚠️ Invalid SMILES: {name}")
            continue

        mol = Chem.AddHs(mol)

        try:
            AllChem.EmbedMolecule(mol)
            AllChem.UFFOptimizeMolecule(mol)
        except:
            continue

        pdb = f"ligands/{safe_name}.pdb"
        pdbqt = f"ligands/{safe_name}.pdbqt"

        Chem.MolToPDBFile(mol, pdb)
        os.system(f"obabel {pdb} -O {pdbqt}")

# =========================
# DOCKING
# =========================
def run_vina(center):

    cx, cy, cz = center

    ligands = [l for l in os.listdir("ligands") if l.endswith(".pdbqt")]

    for lig in tqdm(ligands, desc="🚀 Docking progress"):

        name = lig.replace(".pdbqt", "")

        cmd = (
            f"vina --receptor protein/protein.pdbqt "
            f"--ligand ligands/{lig} "
            f"--center_x {cx} --center_y {cy} --center_z {cz} "
            f"--size_x 20 --size_y 20 --size_z 20 "
            f"--out results/{name}.pdbqt "
            f"--log results/{name}.txt"
        )

        ret = os.system(cmd)
        if ret != 0 :
              print(f"⚠️ Docking failed: {name}")

# =========================
# PARSE RESULTS
# =========================
def parse_results(df):

    data = []

    for f in os.listdir("results"):

        if f.endswith(".txt"):

            # 🔥 FIX: extract correct name
            name = f.replace(".txt", "").replace("_", " ")

            for line in open(f"results/{f}"):

                parts = line.split()

                if len(parts) > 1 and parts[0] == "1":

                    affinity = float(parts[1])

                    data.append([name, affinity])
                    break

    dock = pd.DataFrame(data, columns=["Name", "Docking_Affinity"])

    print("\n🔎 Docking results preview:")
    print(dock.head())

    # 🔥 SAFE MERGE
    merged = df.merge(dock, on="Name", how="left")

    print("\n🔎 After merge check:")
    print("Total rows:", len(merged))
    print("ChEMBL IDs present:", merged["ChEMBL ID"].notna().sum())

    failed = merged[merged["Docking_Affinity"].isna()]
    failed.to_csv("failed_docking.csv", index=False)
    # =========================
    # 🚨 REMOVE FAILED DOCKING
    # =========================
    n_before = len(merged)
    merged = merged.dropna(subset=["Docking_Affinity"])
    n_after = len(merged)
    print(f"⚠ Removed {n_before - n_after} compounds (docking failed)")
    print(f"✔ Remaining valid dockings: {n_after}")

    # Save cleaned results
    merged.to_csv("final_docking_results.csv", index=False)

    print("\n✅ Results saved → final_docking_results.csv")

    return merged

# =========================
# MAIN
# =========================
def main():

    print("\nSTEP 6 — FULLY AUTOMATED DOCKING")

    print("\nSelect execution mode:")
    print("1 → AUTO (Top 50 by Score)")
    print("2 → INTERACTIVE (manual selection)")

    choice = input("Mode: ").strip()

    if choice == "1":
        mode = "auto"
    elif choice == "2":
        mode = "interactive"
    else:
        print("⚠️ Invalid choice → defaulting to INTERACTIVE")
        mode = "interactive"

    reset_workspace()
    df = select_drugs(mode=mode)

    # Save selected drugs
    df.to_csv("selected_compounds.csv", index=False)

    print("\n📄 Saved: selected_compounds.csv")
    print(df.head())

    # =========================
    # TARGET → STRUCTURE
    # =========================
    chembl = load_target()
    uniprot = get_uniprot(chembl)
    pdb_ids = get_pdb_ids(uniprot)
    pdb = select_pdb(pdb_ids)

    print(f"\n🚀 Docking {len(df)} compounds on {pdb}")

    # =========================
    # METADATA
    # =========================
    with open("docking_metadata.json", "w") as f:
        json.dump({
            "chembl": chembl,
            "uniprot": uniprot,
            "pdb": pdb,
            "mode": mode,
            "selection_rule": "Top 50 by Score" if mode == "auto" else "User-defined",
            "n_compounds": len(df)
        }, f, indent=2)

    # =========================
    # PIPELINE
    # =========================
    download_pdb(pdb)
    clean_protein()
    prepare_protein()

    pocket = get_best_pocket()
    center = get_center(pocket)

    prepare_ligands(df)
    run_vina(center)

    results = parse_results(df)

    print("\n🏆 Top docking results:")
    print(results.head())


if __name__ == "__main__":
    main()
