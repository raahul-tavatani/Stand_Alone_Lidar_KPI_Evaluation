# run_kpi_pipeline.py

import argparse
print("Running KPI Evaluation Pipeline...")
from evaluation.generate_target_vicinities import generate_for_all
from evaluation.run_evaluation import run_all_evaluations

def main():
    parser = argparse.ArgumentParser(description="Run KPI Evaluation Pipeline")
    parser.add_argument("--mode", choices=["sensor_data", "simulated_data"], required=True,
                        help="Choose the data source type.")
    parser.add_argument("--root", default="outputs/multidomain_test",
                        help="Path to the root folder containing test data.")
    parser.add_argument("--grade", default="A",
                        help="Target vicinity tolerance grade: 0, A, B, or C (default: 0)")
    args = parser.parse_args()

    if args.mode == "sensor_data":
        print(f"[INFO] Generating target vicinity from folder structure (mode: sensor_data)")
        generate_for_all(args.root, grade=args.grade)

    print("[INFO] Running KPI evaluation...")
    run_all_evaluations(root_dir=args.root)

if __name__ == "__main__":
    main()
