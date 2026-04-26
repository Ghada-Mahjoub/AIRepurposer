# 🧬 AIRepurposer

**Multi-layer AI-driven drug repurposing framework integrating machine learning, structural biology, network analysis, and literature evidence**

---

## 🚀 Overview

AIRepurposer is a comprehensive computational pipeline designed to identify candidate drugs for repurposing using a **multi-layer validation strategy**.

The framework integrates:

* 🤖 Machine Learning (QSAR modeling)
* 🧪 Molecular Docking
* 🧬 Network Biology
* 📚 Literature Mining (PubMed)
* 🧠 Consensus Ranking

---

## 🔬 Pipeline Workflow

The pipeline consists of 9 sequential steps:

### 1. Data Retrieval

Fetch bioactivity data from ChEMBL for a given target.

### 2. Data Preprocessing

Clean, normalize, and convert IC50 values to pIC50.

### 3. Feature Engineering

Generate molecular descriptors and fingerprints.

### 4. Model Training

Train multiple ML models and build an ensemble predictor.

### 5. Drug Screening

Predict activity for DrugBank compounds.

### 6. Structural Validation (Docking + Interaction Analysis)

Step 6 implements a **fully automated structure-based validation pipeline**, integrating molecular docking, interaction analysis, and visualization.

This step consists of multiple submodules:

#### 🔹 6.1 Target Structure Preparation

* Retrieve UniProt ID from ChEMBL
* Identify available PDB structures
* Select best structure (resolution-based)
* Clean and prepare protein

#### 🔹 6.2 Binding Site Detection

* Automatic pocket detection using **fpocket**
* Selection based on druggability score
* Extraction of docking center coordinates

#### 🔹 6.3 Ligand Preparation

* SMILES → 3D structure (RDKit)
* Energy minimization
* Conversion to docking format (PDBQT)

#### 🔹 6.4 Molecular Docking

* Docking performed using **AutoDock Vina**
* Exhaustive search within detected pocket
* Binding affinity (kcal/mol) extracted

#### 🔹 6.5 Docking Analysis

* Merge docking results with ML predictions
* Remove failed docking cases
* Generate final dataset (`final_docking_results.csv`)

#### 🔹 6.6 Statistical & Visual Analysis

* Correlation between ML predictions and docking
* Score distribution plots
* Docking affinity distribution
* Identification of top candidates

#### 🔹 6.7 3D Interaction Visualization

* Interactive protein–ligand visualization using **py3Dmol**
* Pocket-focused rendering
* Export as HTML for publication and sharing

#### 🔹 6.8 Interaction Profiling (PLIP)

* Automated interaction detection:

  * Hydrogen bonds
  * Hydrophobic contacts
  * π–π stacking
* Generation of structural reports and PyMOL sessions

#### 🔹 6.9 Modular Execution

* Full pipeline or individual modules can be executed via control panel

---

📌 This step provides **structural validation of predicted candidates**, ensuring that top-ranked drugs are not only statistically promising but also physically compatible with the target binding site.


### 7. Network Analysis

Evaluate biological relevance using protein interaction networks.

### 8. Literature Validation

Quantify evidence from PubMed using a scoring system.

### 9. Integrated Ranking

Combine all layers into a final consensus score.

---

## ⚙️ Installation

### Clone repository

```bash
git clone https://github.com/Ghada-Mahjoub/AIRepurposer.git
cd AIRepurposer
```

### Create environment

```bash
conda env create -f environment.yml
conda activate airepurposer
```

---

## ▶️ Usage

## ▶️ Usage

Run the pipeline step by step:

```bash
python scripts/01_fetch_chembl_data.py
python scripts/02_data_preprocessing.py
python scripts/03_feature_engineering.py
python scripts/04_model_training.py
python scripts/05_drug_prediction.py
```

### 🔬 Step 6 — Structural Validation

Run the interactive control panel:

```bash
python scripts/06_run_full_step06.py
```

This allows you to choose:

* Full pipeline (Docking → Analysis → Visualization → PLIP)
* Docking only
* Visualization only
* PLIP interaction analysis

Continue with:

```bash
python scripts/07_network_analysis.py
python scripts/08_literature_validation.py
python scripts/09_integrated_ranking.py
```


---

## 📊 Outputs

* Ranked drug candidates (`final_ranked_drugs.csv`)
* Docking visualizations
* Network analysis plots
* Final consensus ranking figure

---

## 📚 Data Sources

* **ChEMBL Database**
  https://www.ebi.ac.uk/chembl/

* **DrugBank**
  Wishart DS et al. *DrugBank 5.0: a major update to the DrugBank database.*
  Nucleic Acids Res. 2018.
  https://go.drugbank.com/

---

## ⚠️ Notes

* DrugBank dataset is used for research purposes only.
* Users should comply with DrugBank licensing terms.

---

## 🔮 Future Work

* Web interface (Streamlit)
* Deep learning models
* Multi-target analysis
* Clinical validation integration...

---

## 👩‍🔬 Author

**Ghada Mahjoub**
PhD Researcher in Bioinformatics__Institut Pasteur Tunis

---

## ⭐ If you find this project useful, consider giving it a star and cite us!
