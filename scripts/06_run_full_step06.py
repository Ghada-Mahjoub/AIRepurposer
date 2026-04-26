import subprocess
import sys

def run_script(script):
    print(f"\n🚀 Running: {script}")
    result = subprocess.run([sys.executable, script])

    if result.returncode != 0:
        print(f"❌ Error in {script}")
        sys.exit(1)

    print(f"✅ Finished: {script}")

def main():
    print("\n========================================")
    print("   STEP 6 — FULL CONTROL PANEL")
    print("========================================")

    print("""
1 → Full pipeline (Docking → Analysis → Visualization → PLIP)
2 → Docking only
3 → Analysis only
4 → Visualization (py3Dmol)
5 → PLIP interaction analysis
6 → Docking + PLIP
7 → Visualization + PLIP (no docking)
""")

    choice = input("Choice: ").strip()

    if choice == "1":
        run_script("06_docking.py")
        run_script("06_analysis.py")
        run_script("06_visualization.py")
        run_script("06_plip_analysis.py")

    elif choice == "2":
        run_script("06_docking.py")

    elif choice == "3":
        run_script("06_analysis.py")

    elif choice == "4":
        run_script("06_visualization.py")

    elif choice == "5":
        run_script("06_plip_analysis.py")

    elif choice == "6":
        run_script("06_docking.py")
        run_script("06_plip_analysis.py")

    elif choice == "7":
        run_script("06_visualization.py")
        run_script("06_plip_analysis.py")

    else:
        print("❌ Invalid choice")
        sys.exit(1)

    print("\n🎉 STEP 6 DONE")
