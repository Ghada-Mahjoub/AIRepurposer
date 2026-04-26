# ==========================================
# INTERACTION VISUALIZATION (py3Dmol - HTML)
# ==========================================

import os
import pandas as pd
import py3Dmol

# ==========================================
# CONFIG
# ==========================================
PROTEIN_FILE = "protein/protein.pdb"
RESULTS_DIR = "results"
OUTPUT_DIR = "interactions"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# CONVERT PDBQT → PDB
# ==========================================
def convert_to_pdb(name):

    pdbqt = os.path.join(RESULTS_DIR, f"{name}.pdbqt")
    pdb = os.path.join(RESULTS_DIR, f"{name}.pdb")

    if not os.path.exists(pdbqt):
        print(f"❌ Missing file: {pdbqt}")
        return None

    os.system(f"obabel {pdbqt} -O {pdb} > /dev/null 2>&1")

    return pdb


# ==========================================
# CREATE HTML VISUALIZATION
# ==========================================
def create_visualization(name):

    print(f"\n🔬 Processing: {name}")

    # =========================
    # Load protein
    # =========================
    if not os.path.exists(PROTEIN_FILE):
        raise RuntimeError("❌ Protein file not found")

    with open(PROTEIN_FILE) as f:
        protein = f.read()

    # =========================
    # Convert ligand
    # =========================
    ligand_file = convert_to_pdb(name)
    if ligand_file is None:
        return

    with open(ligand_file) as f:
        ligand = f.read()

    # =========================
    # Create viewer
    # =========================
    view = py3Dmol.view(width=1000, height=800)

    # =========================
    # Protein 
    # =========================
    view.addModel(protein, "pdb")

    view.setStyle(
            {"model": 0},
            {"cartoon": {"color": "gray", "opacity": 0.45}}
    )

    # =========================
    # Ligand (FOCUS — HIGH CONTRAST)
    # =========================
    view.addModel(ligand, "pdb")

    view.setStyle(
        {"model": 1},
        {
            "stick": {
                "colorscheme": "magentaCarbon",  # 🔥 high contrast color
                "radius": 0.45
            }
        }
    )
    view.addStyle(
           {"within": {"distance": 3.0, "model": 1}},
           {"stick": {"colorscheme": "greenCarbon", "radius": 0.2}}
    )

    # Optional label
    view.addLabel(
        name,
        {
            "position": {"model": 1},
            "backgroundColor": "white",
            "fontColor": "black",
            "fontSize": 12
        }
    )

    # =========================
    # Focus EXACTLY on ligand
    # =========================
    # Balanced view: ligand + nearby pocket
    view.zoomTo({"within": {"distance": 5, "model": 1}})
    view.center({"within": {"distance": 5, "model": 1}})

    # =========================
    # POCKET SURFACE ONLY (STRICT)
    # =========================
    # =========================
    # POCKET SURFACE (MAIN — CLEAN)
    # =========================
    pocket_sel = {"within": {"distance": 3.5, "model": 1}}

    view.addSurface(
       py3Dmol.VDW,
    {
        "opacity": 0.45,          # balanced visibility
        "color": "lightblue"
    },
    pocket_sel
)

   # =========================
   # OPTIONAL: faint protein context (VERY LIGHT)
   # =========================
    view.addSurface(
         py3Dmol.VDW,
    {
        "opacity": 0.05,          
        "color": "gray"
    },
    {"model": 0}
)
    # =========================
    # FINAL ZOOM (tight view)
    # =========================

    view.setBackgroundColor("white")

    # =========================
    # Save HTML
    # =========================
    html_file = os.path.join(OUTPUT_DIR, f"{name}.html")

    with open(html_file, "w") as f:
        f.write(view._make_html())

    print(f"✅ Saved: {html_file}")


# ==========================================
# PROCESS TOP CANDIDATES
# ==========================================
def process_top_candidates(csv_file):

    if not os.path.exists(csv_file):
        raise RuntimeError(f"❌ File not found: {csv_file}")

    df = pd.read_csv(csv_file)

    if "Name" not in df.columns:
        raise ValueError("❌ CSV must contain 'Name' column")

    names = df["Name"].tolist()

    print(f"\n🚀 Generating visualizations for {len(names)} compounds")

    for name in names:
        safe_name = name.replace(" ", "_")
        create_visualization(safe_name)


# ==========================================
# MAIN
# ==========================================
def main():

    print("\n🔬 INTERACTION VISUALIZATION")

    print("\nChoose input:")
    print("1 → top_candidates_clean.csv")
    print("2 → custom CSV")

    choice = input("Choice: ").strip()

    if choice == "1":
        csv_file = "top_candidates_clean.csv"
    elif choice == "2":
        csv_file = input("Enter CSV path: ").strip()
    else:
        raise ValueError("Invalid choice")

    process_top_candidates(csv_file)

    print("\n✅ All visualizations generated!")
    print(f"📁 Open HTML files in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
