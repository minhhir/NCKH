import pandas as pd
from Analysis.Analysis import run_analysis


JOBS = [
    {
        "input_csv": "synthetic_data_hypothesis_optimized.csv",
        "with_trust": "Synthetic_Results_Optimized_With_Trust.txt",
        "without_trust": "Synthetic_Results_Optimized_Without_Trust.txt",
        "label": "synthetic_data_hypothesis_optimized.csv",
    },
]


def run_job(job):
    print(f"\n===== GENERATE RESULTS FOR: {job['label']} =====")
    df = pd.read_csv(job["input_csv"])

    run_analysis(
        df,
        include_trust=True,
        output_file=job["with_trust"],
    )
    run_analysis(
        df,
        include_trust=False,
        output_file=job["without_trust"],
    )

    print(f"✓ Created: {job['with_trust']}")
    print(f"✓ Created: {job['without_trust']}")


def main():
    for job in JOBS:
        try:
            run_job(job)
        except FileNotFoundError:
            print(f"⚠ Skip: cannot find {job['input_csv']}")
        except Exception as error:
            print(f"✗ Failed for {job['input_csv']}: {error}")


if __name__ == "__main__":
    main()
