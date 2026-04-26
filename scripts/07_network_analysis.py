"""
STEP 7 — NETWORK VALIDATION (FINAL — PUBLICATION LEVEL)

✔ Fix UniProt ↔ Gene mismatch
✔ Real targets from ChEMBL
✔ STRING network (expanded)
✔ Centrality + hub detection
✔ KEGG enrichment (biological meaning)
✔ Drug-network scoring (non-zero)
✔ Publication-ready figures
✔ Cytoscape export

Author: Ghada Mahjoub
"""

import json
import requests
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from gprofiler import GProfiler
import os
import time
import hashlib
import requests


CACHE_DIR = "api_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def cache_path(url):
    key = hashlib.md5(url.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{key}.json")

STRING_API = "https://string-db.org/api/tsv/network"
SPECIES = 9606
# =========================
# GENE NORMALIZATION 
# =========================
def normalize_gene(g):

    if not isinstance(g, str):
        return None

    return g.upper().replace("_HUMAN", "").strip()
# =========================
# SAFE REQUEST
# =========================
def safe_request(url, use_cache=True, retries=3):

    path = cache_path(url)

    # ✅ Load from cache
    if use_cache and os.path.exists(path):
        with open(path) as f:
            return json.load(f)

    # 🌐 Try API
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30)

            if r.status_code == 200:
                data = r.json()

                # 💾 Save cache
                with open(path, "w") as f:
                    json.dump(data, f)

                return data

            else:
                print(f"⚠️ HTTP {r.status_code} ({attempt+1})")

        except Exception as e:
            print(f"⚠️ Attempt {attempt+1} failed: {e}")

        time.sleep(2)

    print(f"❌ Failed request: {url}")
    return None


# =========================
# LOAD TARGET
# =========================
def load_target():
    with open("project_config.json") as f:
        return json.load(f)["target_chembl_id"]


# =========================
# LOAD DOCKED DRUGS
# =========================
def load_drugs():

    df = pd.read_csv("final_docking_results.csv")
    df.columns = df.columns.str.strip()

    df["ChEMBL ID"] = df["ChEMBL ID"].astype(str).str.strip()

    df = df[df["ChEMBL ID"].str.startswith("CHEMBL")]

    # Combined score
    df["Docking_norm"] = -df["Docking_Affinity"]
    df["Combined"] = 0.6 * df["Docking_norm"] + 0.4 * df["Score"]

    df = df.sort_values("Combined", ascending=False).head(10)

    return df


# =========================
# GET TARGETS (ChEMBL)
# =========================
def get_targets(chembl_id):

    import numpy as np

    url = f"https://www.ebi.ac.uk/chembl/api/data/activity.json?molecule_chembl_id={chembl_id}&limit=100"

    print(f"🔎 Fetching targets for {chembl_id}")

    data = safe_request(url)

    if not data:
        print(f"⚠️ No data for {chembl_id}")
        return []

    activities = data.get("activities", [])

    if not activities:
        print(f"⚠️ No activities for {chembl_id}")
        return []

    # 🔥 Store targets with their pChEMBL values
    targets = {}

    for act in activities:
        tid = act.get("target_chembl_id")
        organism = act.get("target_organism")
        pchembl = act.get("pchembl_value")

        # ✅ FILTER: valid + human + has potency
        if (
            tid
            and tid.startswith("CHEMBL")
            and organism == "Homo sapiens"
            and pchembl is not None
        ):
            try:
                pchembl = float(pchembl)

                # ✅ Keep only strong interactions (IC50 ≤ ~1 µM)
                if pchembl >= 6:
                    if tid not in targets:
                        targets[tid] = []
                    targets[tid].append(pchembl)

            except:
                continue

    if not targets:
        print(f"⚠️ {chembl_id}: no strong HUMAN targets after filtering")
        return []

    # 🔥 Rank targets by mean potency
    ranked = sorted(
        targets.items(),
        key=lambda x: np.mean(x[1]),
        reverse=True
    )

    # ✅ Keep top 10 biologically relevant targets
    top_targets = [t[0] for t in ranked[:10]]

    print(f"✅ {chembl_id}: {len(top_targets)} filtered HUMAN targets")

    return top_targets

# ==============================
# UniProt → Gene Symbol Mapping
# ==============================
def uniprot_to_gene(uniprot_ids):
    import requests
    
    if not uniprot_ids:
        return {}

    url = "https://rest.uniprot.org/uniprotkb/stream"
    query = " OR ".join([f"accession:{u}" for u in uniprot_ids])

    params = {
        "query": query,
        "format": "tsv",
        "fields": "accession,gene_primary"
    }

    r = requests.get(url, params=params)

    mapping = {}
    if r.status_code == 200:
        lines = r.text.split("\n")[1:]
        for line in lines:
            parts = line.split("\t")
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1]

    return mapping
# =========================
# MAP TARGET → GENE
# =========================
def target_to_gene(tid):

    url = f"https://www.ebi.ac.uk/chembl/api/data/target/{tid}.json"
    data = safe_request(url)

    if not data:
        return None

    comps = data.get("target_components", [])
    if not comps:
        return None

    comp = comps[0]

    # 1️⃣ Direct gene name (best case)
    gene = comp.get("gene_name")
    if gene:
        return normalize_gene(gene)

    # 2️⃣ HGNC mapping
    for x in comp.get("target_component_xrefs", []):
        if x.get("xref_src_db") == "HGNC":
            return normalize_gene(x.get("xref_name"))

    # 3️⃣ Synonyms fallback
    for s in comp.get("target_component_synonyms", []):
        if s.get("syn_type") == "GENE_SYMBOL":
            return normalize_gene(s.get("component_synonym"))

    return None

# =========================
# BUILD DRUG TARGET MAP
# =========================
def build_target_map(df):

    mapping = {}

    for _, row in df.iterrows():

        chembl = row["ChEMBL ID"]
        name = row["Name"]
        tids = get_targets(chembl)
        genes = []   

        for t in tids:
            g = target_to_gene(t)
            if g:
                genes.append(normalize_gene(g))

        # remove None + duplicates
        genes = list(set([g for g in genes if g is not None]))
        if len(genes) == 0:
            print(f"⚠️ {name} has no mapped human targets — excluded from network scoring")

        print(f"{name}: {len(genes)} targets → {genes}")

        mapping[name] = genes

    return mapping


# =========================
# GET MAIN TARGET GENE
# =========================
def get_main_target(chembl_id):

    url = f"https://www.ebi.ac.uk/chembl/api/data/target/{chembl_id}.json"

    data = safe_request(url)

    if not data:
        return None

    try:
        gene = data["target_components"][0].get("gene_name")
        if gene:
            return normalize_gene(gene)

    except:
        pass

    return None


# =========================
# STRING NETWORK
# =========================
def get_string(genes):

    print("\n🔬 STRING network...")

    params = {
        "identifiers": "%0d".join(genes),
        "species": SPECIES,
        "add_nodes": 50
    }

    r = requests.post(STRING_API, data=params)

    edges = []

    for line in r.text.split("\n")[1:]:

        parts = line.split("\t")

        if len(parts) < 6:
            continue

        g1 = parts[2]
        g2 = parts[3]
        score = float(parts[5])

        edges.append((g1, g2, score))

    df = pd.DataFrame(edges, columns=["A", "B", "Score"])

    return df


# =========================
# BUILD GRAPH
# =========================
def build_graph(df):

    G = nx.Graph()

    for _, r in df.iterrows():

        g1 = normalize_gene(r["A"])
        g2 = normalize_gene(r["B"])

        G.add_edge(g1, g2, weight=r["Score"])

    print(f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")

    return G


# =========================
# HUB DETECTION
# =========================
def get_hubs(G):

    deg = nx.degree_centrality(G)

    hubs = sorted(deg, key=deg.get, reverse=True)[:10]

    print("\n🔥 Hub genes:", hubs)

    return hubs, deg


# =========================
# DRUG SCORING
# =========================
def score_drugs(G, targets):

    centrality = nx.betweenness_centrality(G)

    results = []

    for drug, genes in targets.items():

        mapped = [g for g in genes if g in G.nodes()]

        if not mapped:
            score = 0
        else:
            score = np.sum([centrality[g] for g in mapped]) / (len(mapped) + 1)

        results.append([drug, score])

    df = pd.DataFrame(results, columns=["Drug", "Network_Score"])

    return df.sort_values("Network_Score", ascending=False)


# =========================
# KEGG ENRICHMENT
# =========================
def run_kegg(genes):

    gp = GProfiler(return_dataframe=True)

    res = gp.profile(organism="hsapiens", query=genes)

    res = res[(res["source"] == "KEGG") & (res["p_value"] < 0.05)]

    return res.sort_values("p_value")


# =========================
# PLOT NETWORK (CLEAN)
# =========================
def plot_network(G, hubs, main_target=None):

    plt.figure(figsize=(11, 9))  # bigger = publication quality

    # =========================
    # FILTER NETWORK (clean)
    # =========================
    edges = [(u, v) for u, v, d in G.edges(data=True)
             if d.get("weight", 0) > 0.90]

    H = nx.Graph()
    H.add_edges_from(edges)

    # Remove isolated / weak nodes
    H = H.subgraph([
        n for n in H.nodes()
        if H.degree(n) > 1 or n in hubs or n == main_target
    ])

    # Layout
    pos = nx.spring_layout(H, k=0.9, iterations=200, seed=42)

    # Centrality
    centrality = nx.degree_centrality(H)

    # =========================
    # NODE STYLE
    # =========================
    node_sizes = [300 + 2500 * centrality.get(n, 0) for n in H.nodes()]

    node_colors = []
    for n in H.nodes():
        if n == main_target:
            node_colors.append("purple")   # main target
        elif n in hubs:
            node_colors.append("red")      # hubs
        elif centrality.get(n, 0) > np.percentile(list(centrality.values()), 75):
            node_colors.append("orange")   # high centrality
        else:
            node_colors.append("skyblue")  # peripheral

    # =========================
    # DRAW
    # =========================
    nx.draw_networkx_edges(H, pos, alpha=0.3, width=1.2)

    nx.draw_networkx_nodes(
        H, pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="black",
        linewidths=0.4
    )

    # Label ONLY hubs + main target
    labels = {n: n for n in H.nodes() if n in hubs or n == main_target}

    nx.draw_networkx_labels(
        H, pos,
        labels=labels,
        font_size=9,
        font_weight="bold"
    )

    # =========================
    # TITLE + SUBTITLE
    # =========================
    plt.title(
        "Protein–Protein Interaction Network Highlighting Key Targets",
        fontsize=16,
        fontweight="bold"
    )

    plt.figtext(
        0.5, 0.02,
        "Node size reflects degree centrality; edges filtered at STRING score > 0.9",
        ha="center",
        fontsize=10
    )

    # =========================
    # LEGEND
    # =========================
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Primary target',
               markerfacecolor='purple', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Hub genes',
               markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='High centrality',
               markerfacecolor='orange', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Peripheral genes',
               markerfacecolor='skyblue', markersize=10)
    ]

    plt.legend(handles=legend_elements, loc='lower left', frameon=False)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig("Figure_3A_Network.png", dpi=600)
    plt.savefig("Figure_3A_Network.svg")
    plt.close()


# =========================
# PLOT KEGG
# =========================
import textwrap

def plot_kegg(results):

    if results.empty:
        print("⚠️ No KEGG results")
        return

    results = results.sort_values("p_value").head(15)

    results["logP"] = -np.log10(results["p_value"])
    results["GeneRatio"] = results["intersection_size"] / results["query_size"]

    results["name"] = results["name"].apply(
        lambda x: "\n".join(textwrap.wrap(x, 28))
    )

    results = results.iloc[::-1]

    plt.figure(figsize=(10, 7))

    scatter = plt.scatter(
        results["GeneRatio"],
        results["name"],
        s=results["intersection_size"] * 18,
        c=results["logP"],
        cmap="plasma",
        edgecolor="black",
        linewidth=0.5
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label("-log10(adj. p-value)", fontsize=11)

    plt.xlabel("Gene Ratio", fontsize=12)
    plt.title("KEGG Pathway Enrichment", fontsize=16, fontweight="bold")

    plt.grid(axis='x', linestyle='--', alpha=0.3)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for size in [5, 10, 20]:
        plt.scatter([], [], s=size*18, c='gray', label=f"{size} genes")
    plt.legend(title="Gene count", loc='lower right', frameon=False)

    plt.tight_layout()
    plt.savefig("Figure_3B_KEGG_dotplot.png", dpi=600)
    plt.close()
    
# =========================
# DRUG IMPACT PLOT
# =========================
def plot_drug_scores(df):

    if df["Network_Score"].sum() == 0:
        print("⚠️ All scores are zero")
        return

    # Clean + sort (REFERENCE DATAFRAME)
    df = df[df["Network_Score"] > 0].copy()
    df = df.sort_values("Network_Score", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    # =========================
    # IDENTIFY TOP DRUGS (CORRECT WAY)
    # =========================
    top_drugs = df.nlargest(3, "Network_Score")["Drug"].tolist()

    # =========================
    # COLOR ASSIGNMENT
    # =========================
    colors = [
        "#D7263D" if d in top_drugs else "#D3D3D3"
        for d in df["Drug"]
    ]

    # =========================
    # PLOT
    # =========================
    bars = ax.barh(
        df["Drug"],
        df["Network_Score"],
        color=colors,
        edgecolor="black",
        linewidth=0.5
    )

    # =========================
    # VALUES
    # =========================
    for i, v in enumerate(df["Network_Score"]):
        ax.text(v + 0.002, i, f"{v:.3f}", va='center', fontsize=9)

    # =========================
    # OPTIONAL STAR (SAFE POSITION)
    # =========================
    for i, drug in enumerate(df["Drug"]):
        if drug in top_drugs:
            ax.text(
                df["Network_Score"].iloc[i] * 0.95,
                i,
                "★",
                color="white",
                fontsize=10,
                va="center",
                ha="right"
            )

    # =========================
    # MEAN LINE
    # =========================
    mean_score = df["Network_Score"].mean()
    ax.axvline(mean_score, linestyle="--", color="gray", alpha=0.7)

    # Better placement (no overlap)
    ax.text(
        mean_score,
        len(df) - 0.5,
        "Mean",
        ha="center",
        va="bottom",
        fontsize=9,
        color="gray",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
    )

    # =========================
    # LABELS
    # =========================
    ax.set_xlabel("Network Impact Score", fontsize=12)

    ax.set_title(
        "Drug Network Impact Scores",
        fontsize=15,
        fontweight="bold"
    )

    # =========================
    # LEGEND (VERY IMPORTANT)
    # =========================
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor='#D7263D', edgecolor='black', label='Top network-impact drugs'),
        Patch(facecolor='#D3D3D3', edgecolor='black', label='Other drugs')
    ]

    ax.legend(handles=legend_elements, frameon=False, loc='lower right')

    # =========================
    # STYLE
    # =========================
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig("Figure_3C_Drug.png", dpi=600)
    plt.savefig("Figure_3C_Drug.svg")
    plt.close()

# =========================
# MULTI-PANEL FIGURE
# =========================
def plot_final_figure(G, hubs, kegg_results, drug_scores):

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1])

    # ======================
    # A — NETWORK
    # ======================
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_anchor('C')
    edges = [(u, v) for u, v, d in G.edges(data=True)
             if d.get("weight", 0) > 0.85]

    H = nx.Graph()
    H.add_edges_from(edges)
    if len(H.nodes()) == 0:
        print("⚠️ Empty network — skipping plot")
        return

    # Clean network
    H = H.subgraph([
        n for n in H.nodes()
        if H.degree(n) > 1 or n in hubs
    ])

    pos = nx.spring_layout(H, k=0.9, iterations=200, seed=42)
    centrality = nx.degree_centrality(H)

    node_sizes = [150 + 1500 * centrality.get(n, 0) for n in H.nodes()]

    node_colors = [
        "red" if n in hubs else
        "orange" if centrality.get(n, 0) > np.percentile(list(centrality.values()), 75)
        else "skyblue"
        for n in H.nodes()
    ]
     
    nx.draw_networkx_edges(H, pos, ax=ax1, alpha=0.2, width=1)
    nx.draw_networkx_nodes(
        H, pos, ax=ax1,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="black",
        linewidths=0.3
    )

    hub_labels = {n: n for n in hubs if n in H.nodes()}
    nx.draw_networkx_labels(H, pos, labels=hub_labels, ax=ax1, font_size=9)
    
    # shift view (down + right effect)
    x_vals = [p[0] for p in pos.values()]
    y_vals = [p[1] for p in pos.values()]

    ax1.set_xlim(min(x_vals) - 0.1, max(x_vals) + 0.3)  # push right
    ax1.set_ylim(min(y_vals) - 0.3, max(y_vals) + 0.1)  # push down
    
    from matplotlib.lines import Line2D
    legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Hub genes',
                          markerfacecolor='red', markersize=8, markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', label='High centrality',
                          markerfacecolor='orange', markersize=8, markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', label='Peripheral genes',
                          markerfacecolor='skyblue', markersize=8, markeredgecolor='black')
   ]

    ax1.legend(
            handles=legend_elements,
            loc='upper left',
            frameon=False,
            fontsize=9
    )
    ax1.set_title("A. Protein Interaction Network", loc="left", fontweight="bold", fontsize=14)
    ax1.axis("off")

    # ======================
    # B — KEGG
    # ======================
    ax2 = fig.add_subplot(gs[1, 0])

    kegg = kegg_results.sort_values("p_value").head(10).copy()
    kegg["logP"] = -np.log10(kegg["p_value"])
    kegg["GeneRatio"] = kegg["intersection_size"] / kegg["query_size"]
    kegg = kegg.iloc[::-1]

    sc = ax2.scatter(
        kegg["GeneRatio"],
        kegg["name"],
        s=kegg["intersection_size"] * 20,
        c=kegg["logP"],
        cmap="plasma",
        edgecolor="black"
    )

    cbar = fig.colorbar(sc, ax=ax2)
    cbar.set_label("-log10(adj. p-value)", fontsize=10)

    ax2.set_title("B.KEGG Pathway Enrichment", loc="left", fontweight="bold", fontsize=14)
    ax2.set_xlabel("Gene Ratio")
    ax2.set_ylabel("")
    ax2.tick_params(axis='y', labelsize=9)

    # ======================
    # C — DRUG
    # ======================
    ax3 = fig.add_subplot(gs[1, 1])

    df = drug_scores.copy()
    df = df[df["Network_Score"] > 0]
    df = df.sort_values("Network_Score", ascending=True).reset_index(drop=True)

    # Identify top drugs correctly
    top_drugs = df.nlargest(3, "Network_Score")["Drug"].tolist()

    colors = [
        "#d62728" if d in top_drugs else "#c7c7c7"
        for d in df["Drug"]
    ]

    bars = ax3.barh(
        df["Drug"],
        df["Network_Score"],
        color=colors,
        edgecolor="black",
        linewidth=0.6
    )

    # Values
    for i, v in enumerate(df["Network_Score"]):
        ax3.text(v + 0.001, i, f"{v:.3f}", va='center', fontsize=9)

    # Mean line
    mean_score = df["Network_Score"].mean()
    ax3.axvline(mean_score, linestyle="--", color="gray", alpha=0.7)

    ax3.text(
        mean_score,
        len(df)  -0.5,
        "Mean",
        ha="center",
        va="bottom",
        fontsize=9,
        color="gray",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")  
    )

    ax3.set_title("C.Drug Network Impact Scores", loc="left", fontweight="bold", fontsize=14)
    ax3.set_xlabel("Network Impact Score")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d62728", label="Top network-impact drugs"),
        Patch(facecolor="#c7c7c7", label="Other drugs")
    ]
    ax3.legend(handles=legend_elements, frameon=False, fontsize=9)

    # ======================
    # GLOBAL TITLE
    # ======================
    fig.suptitle(
        "Network-Based Drug Repurposing Reveals Key Targets and Candidate Compounds",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig("Figure_3_MultiPanel.png", dpi=600)
    plt.savefig("Figure_3_MultiPanel.svg")
    plt.close()
# =========================
# CYTOSCAPE EXPORT
# =========================
def export_graph(G, hubs):

    degree = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)

    for node in G.nodes():
        G.nodes[node]["degree"] = degree.get(node, 0)
        G.nodes[node]["betweenness"] = betweenness.get(node, 0)
        G.nodes[node]["is_hub"] = 1 if node in hubs else 0
        G.nodes[node]["label"] = node
        G.nodes[node]["type"] = "hub" if node in hubs else "non-hub"

    for u, v, d in G.edges(data=True):
        d["weight"] = d.get("weight", 0)

    nx.write_graphml(G, "network.graphml")


# =========================
# =========================
# MAIN
# =========================
def main():

    print("\n=== Step 7: Network validation ===")

    # ----------------------
    # Load target + drugs
    # ----------------------
    target = load_target()
    print("🎯 Target:", target)

    df = load_drugs()

    print(f"\n📊 Selected {len(df)} drugs")

    # ----------------------
    # Extract drug targets (CORRECT)
    # ----------------------
    drug_targets = build_target_map(df)

    # ----------------------
    # Add main target gene
    # ----------------------
    target_gene = get_main_target(target)

    # Build gene list
    all_genes = list(set(
       sum(drug_targets.values(), []) +
       ([target_gene] if target_gene else [])
    ))

    # 🔧 CLEAN GENES (VERY IMPORTANT)
    def is_valid_gene(g):
        return (
            isinstance(g, str)
            and g.isupper()
            and 2 <= len(g) <= 12
        )

    all_genes = [g for g in all_genes if is_valid_gene(g)]

    print("\n🧬 Clean gene set:", len(all_genes))
    print(all_genes[:10])

    if not all_genes:
        raise ValueError("❌ No genes available")
    if len(all_genes) < 5:
        print("⚠️ Too few genes — network may be unreliable")
    print(f"\n🧬 Final gene set: {len(all_genes)} genes")

    # ----------------------
    # STRING network
    # ----------------------
    print("\n🔬 Fetching STRING network...")

    string_df = get_string(all_genes)

    G = build_graph(string_df)

    print(f"📊 Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")
   # ----------------------
   # Check overlap (DEBUG)
   # ----------------------
    print("\n🔍 Checking overlap between targets and network nodes...")

    for drug, genes in drug_targets.items():
        overlap = [g for g in genes if g in G.nodes()]
        print(f"{drug}: {len(overlap)} overlap → {overlap}")
    
    print("\n🧬 Example mapped genes:")
    for d, g in list(drug_targets.items())[:3]:
        print(d, "→", g)
    # ----------------------
    # Network analysis
    # ----------------------
    hubs, deg = get_hubs(G)

    # ----------------------
    # Drug scoring
    # ----------------------
    print("\n💊 Scoring drugs...")

    scores = score_drugs(G, drug_targets)
    scores["Network_Score_norm"] = (
            scores["Network_Score"] - scores["Network_Score"].min()
    ) / (scores["Network_Score"].max() - scores["Network_Score"].min()+ 1e-9)

    scores.to_csv("drug_network_scores.csv", index=False)
    if scores.empty:
        print("⚠️ No valid scores — skipping plot")
        return

    # ----------------------
    # KEGG enrichment
    # ----------------------
    print("\n🧬 KEGG enrichment...")

    kegg = run_kegg(all_genes)

    kegg.to_csv("kegg.csv", index=False)

    # ----------------------
    # Plots
    # ----------------------
    print("\n📊 Generating figures...")

    plot_network(G, hubs)
    plot_kegg(kegg)
    plot_drug_scores(scores)
    plot_final_figure(G, hubs, kegg, scores)
    # ----------------------
    # Export
    # ----------------------
    export_graph(G,hubs)

    print("\n🎯 Step 7 DONE — next!")


if __name__ == "__main__":
    main()
