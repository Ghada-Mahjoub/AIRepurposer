"""
AIRepurposer - Step 1 (FINAL STABLE VERSION)

✔ Hybrid search (client + REST + manual fallback)
✔ Input validation (CRITICAL)
✔ Safe API calls
✔ Retry logic
✔ Partial backup
✔ Metadata saving

Author: Ghada Mahjoub
"""

import requests
import pandas as pd
import time
import json

# =========================
# 🔹 SAFE REQUEST
# =========================
def safe_request(url, params=None, retries=5):

    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)

            if r.status_code != 200:
                print(f"⚠️ Server error {r.status_code} (try {i+1})")
                time.sleep(2)
                continue

            return r.json()

        except Exception as e:
            print(f"⚠️ Error: {e} (try {i+1})")
            time.sleep(2)

    return None


# =========================
# 🔍 ROBUST SEARCH
# =========================
def search_target(query):

    print("\n🔎 Searching targets...")

    # 1️⃣ Try client
    try:
        from chembl_webresource_client.new_client import new_client

        target = new_client.target
        results = target.search(query)

        df = pd.DataFrame(results)

        if not df.empty:
            df = df[['target_chembl_id', 'pref_name', 'organism']]
            print("\n📋 Results (client):\n")
            print(df.head(10).to_string(index=False))
            return df

    except Exception as e:
        print(f"⚠️ Client failed: {e}")

    # 2️⃣ REST fallback
    try:
        url = "https://www.ebi.ac.uk/chembl/api/data/target/search"
        params = {"q": query, "format": "json"}

        data = safe_request(url, params)

        if data:
            df = pd.DataFrame(data.get("targets", []))

            if not df.empty:
                df = df[['target_chembl_id', 'pref_name', 'organism']]
                print("\n📋 Results (REST fallback):\n")
                print(df.head(10).to_string(index=False))
                return df

    except Exception as e:
        print(f"⚠️ REST fallback failed: {e}")

    print("\n❌ Search failed.")
    return None


# =========================
# 🔹 FETCH DATA
# =========================
def fetch_data(target_id, activity="IC50", limit=1000):

    print("\n⏳ Fetching bioactivity data...")

    base_url = "https://www.ebi.ac.uk/chembl/api/data/activity"

    offset = 0
    all_data = []
    failures = 0
    page_size = 200

    while len(all_data) < limit:

        params = {
            "target_chembl_id": target_id.upper(),
            "standard_type": activity.upper(),
            "limit": page_size,
            "offset": offset,
            "format": "json"
        }

        data = safe_request(base_url, params)

        if not data:
            failures += 1
            print(f"⚠️ Failed request ({failures}/10)")

            if failures > 10:
                print("❌ Too many failures. Stopping.")
                break

            continue

        failures = 0

        activities = data.get("activities", [])

        if not activities:
            print("✔ End of data reached")
            break

        all_data.extend(activities)

        print(f"📦 Retrieved {len(all_data)} records...")

        # backup
        if len(all_data) % 500 == 0:
            pd.DataFrame(all_data).to_csv("partial_backup.csv", index=False)
            print("💾 Partial backup saved")

        if len(activities) < page_size:
            print("✔ Last page reached")
            break

        offset += page_size
        time.sleep(0.3)

    df = pd.DataFrame(all_data)

    if df.empty:
        print("❌ No data retrieved")
        return None

    df = df[[
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_value",
        "standard_units"
    ]].dropna()

    print(f"\n✅ Final usable records: {len(df)}")

    return df


# =========================
# 🔹 MAIN
# =========================
def run_fetch():

    print("\n=== Step 1: ChEMBL Retrieval (FINAL) ===")

    # MODE
    while True:
        mode = input("1-ID / 2-Name: ").strip()
        if mode in ["1", "2"]:
            break
        print("⚠️ Please enter 1 or 2")

    # =====================
    # MODE 2: SEARCH
    # =====================
    if mode == "2":

        query = input("Enter target name (e.g., acetylcholinesterase): ").strip()

        df_targets = search_target(query)

        if df_targets is None:
            print("\n⚠️ Switching to manual mode")

            while True:
                target_id = input("Enter ChEMBL ID: ").strip().upper()
                if target_id.startswith("CHEMBL"):
                    break
                print("⚠️ Invalid ID (example: CHEMBL220)")

        else:
            while True:
                target_id = input("\nEnter selected ChEMBL ID: ").strip().upper()
                if target_id.startswith("CHEMBL"):
                    break
                print("⚠️ Invalid ID")

    # =====================
    # MODE 1: DIRECT
    # =====================
    else:
        while True:
            target_id = input("Enter ChEMBL ID (e.g., CHEMBL203): ").strip().upper()
            if target_id.startswith("CHEMBL"):
                break
            print("⚠️ Invalid ID (example: CHEMBL220)")

    print(f"\n🎯 Target selected: {target_id}")

    # =====================
    # ACTIVITY
    # =====================
    valid_types = ["IC50", "KI", "EC50"]

    while True:
        activity = input("Activity (IC50/Ki/EC50) [IC50]: ").strip().upper()

        if activity == "":
            activity = "IC50"
            break
        elif activity in valid_types:
            break
        else:
            print("⚠️ Choose IC50, Ki or EC50")

    # =====================
    # LIMIT
    # =====================
    while True:
        limit = input("Max records [e.g,1000]: ").strip()

        if limit == "":
            limit = 1000
            break
        elif limit.isdigit():
            limit = int(limit)
            break
        else:
            print("⚠️ Enter a valid number")

    # =====================
    # FETCH
    # =====================
    df = fetch_data(target_id, activity, limit)

    if df is None:
        return

    print("\n📊 Dataset preview:")
    print(df.head())

    filename = f"chembl_{target_id}_{activity}.csv"
    df.to_csv(filename, index=False)

    print(f"\n💾 Saved: {filename}")

    # =====================
    # SAVE METADATA 
    # =====================
    print("\n🧠 Extracting known active compounds...")

    known_compounds = df["molecule_chembl_id"].dropna().unique().tolist()
    disease = input("Enter disease (e.g., cancer, bacterial infection): ").strip()

    meta = {
        "target_chembl_id": target_id,
        "target_name": target_name,
        "activity_type": activity,
        "known_active_compounds": known_compounds,
        "n_known_compounds": len(known_compounds),
        "disease": disease,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open("project_config.json", "w") as f:
        json.dump(meta, f, indent=4)

    print(f"✔ Known compounds saved: {len(known_compounds)}")
    print("🧠 Metadata saved (project_config.json)")

    print("\n🎯 Step 1 completed successfully!")
    

# =========================
# ▶ RUN
# =========================
if __name__ == "__main__":
    run_fetch()
