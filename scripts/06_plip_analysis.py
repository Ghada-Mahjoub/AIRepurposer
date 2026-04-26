# ==========================================
# STEP 6 — PLIP INTERACTION ANALYSIS (FIXED)
# ==========================================

import os
import subprocess

RESULTS_DIR = "results"
PROTEIN_FILE = "protein/protein.pdb"
PLIP_DIR = "plip_results"

os.makedirs(PLIP_DIR, exist_ok=True)


# ==========================================
# MERGE PROTEIN + LIGAND
# ==========================================
def merge_complex(protein_pdb, ligand_pdb, output_pdb):

    with open(output_pdb, "w") as out:
        with open(protein_pdb) as f:
            out.writelines(f.readlines())

        with open(ligand_pdb) as f:
            out.writelines(f.readlines())


# ==========================================
# RUN PLIP
# ==========================================
def run_plip(name):

    ligand_pdb = os.path.join(RESULTS_DIR, f"{name}.pdb")

    if not os.path.exists(ligand_pdb):
        print(f"⚠️ Skipping (no PDB): {name}")
        return

    complex_pdb = os.path.join(PLIP_DIR, f"{name}_complex.pdb")
    out_dir = os.path.join(PLIP_DIR, name)

    os.makedirs(out_dir, exist_ok=True)

    merge_complex(PROTEIN_FILE, ligand_pdb, complex_pdb)

    print(f"\n🔬 Running PLIP for: {name}")

    # 🔥 IMPORTANT: force full outputs + pymol
    cmd = [
        "plip",
        "-f", complex_pdb,
        "-o", out_dir,
        "-x",        # XML
        "-t",        # TXT
        "--pymol"    # force PyMOL session
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(result.stderr)

    if result.returncode != 0:
        print(f"❌ PLIP failed: {name}")
    else:
        print(f"✅ PLIP done: {name}")


# ==========================================
# MAIN
# ==========================================
def main():

    print("\n🧬 STEP 6 — PLIP INTERACTION ANALYSIS")

    if not os.path.exists(RESULTS_DIR):
        raise RuntimeError("❌ results/ folder not found")

    ligands = [
        f.replace(".pdbqt", "")
        for f in os.listdir(RESULTS_DIR)
        if f.endswith(".pdbqt")
    ]

    print(f"\n🔍 Found {len(ligands)} ligands")

    for lig in ligands:
        run_plip(lig)

    print("\n🎉 PLIP ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
