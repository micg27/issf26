import argparse
import subprocess
import sys
from pathlib import Path

import config

HERE = Path(__file__).resolve().parent

STAGES = [
    ("02", "02_preprocess_hourly.py", True),
    ("03", "03_qc_hourly.py", True),
    ("04", "04_build_daily_features.py", False),
    ("05", "05_cluster_sweep.py", False),
    ("06", "06_fit_final_clusters.py", False),
    ("07", "07_characterize_regimes.py", False),
    ("09", "09_split_train_test.py", False),
    ("08", "08_select_days.py", False),
    ("10", "10_build_phase3_days.py", False),
    ("11", "11_extract_phase3_data.py", False),
]


def run(script, cityargs):

    command = [sys.executable, str(HERE / script)] + cityargs

    print(f"\n>>> {' '.join(command)}")

    result = subprocess.run(command)

    if result.returncode != 0:
        raise SystemExit(
            f"\nThis stuff does NOT Work. "
        )


def main():

    parser = argparse.ArgumentParser(description="Run the Phase 2 pipeline.")

    parser.add_argument("--city", choices=config.ALL_CITIES)
    parser.add_argument("--from", dest="start", default="02")
    parser.add_argument("--only", default=None)

    args = parser.parse_args()

    cityargs = ["--city", args.city] if args.city else ["--all"]

    cities = [args.city] if args.city else config.ALL_CITIES

    if args.only:
        stages = [s for s in STAGES if s[0] == args.only]
        if not stages:
            raise SystemExit(f"Unknown stage: {args.only}")
    else:
        inds = [i for i, s in enumerate(STAGES) if s[0] == args.start]
        if not inds:
            raise SystemExit(f"Unknown stage: {args.start}")
        stages = STAGES[inds[0]:]

    for code, script, _ in stages:

        # Gate: do not proceed past the sweep until k is chosen.
        if code == "06":

            unset = [
                city for city in cities
                if config.CITIES[city]["final_k"] is None
            ]

            if unset:
                print("\n" + "=" * 68)
                print("STOPPING BEFORE STAGE 06")
                print("=" * 68)
                print(
                    f"\nfinal_k is not set for: {', '.join(unset)}"
                )
                print(
                    "\nInspect outputs/meteorology/<city>/clustering/"
                    "cluster_validity.png"
                )
                print(
                    "and the candidate_k*.csv files, choose k, set it in "
                    "config.py,"
                )
                print("then re-run:  python run_phase2.py --from 06")
                return

        extra = ["--include-representative"] if code == "10" else []
        run(script, cityargs + extra)

    print("\n" + "=" * 68)
    print("PHASE 2 COMPLETE")
    print("=" * 68)
    print(
        "\nPhase 3 inputs are in "
        "data/processed/meteorology/<city>/phase3_weather/"
    )


if __name__ == "__main__":
    main()
