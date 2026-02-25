import numpy as np
import pandas as pd

from Analysis.Analysis import run_analysis
from Visualization.Visualization import visualize_results


def _sigmoid(value):
    return 1.0 / (1.0 + np.exp(-value))


def build_synthetic_dataset(base_file="final_data.csv", seed=42):
    np.random.seed(seed)
    df = pd.read_csv(base_file).copy()

    user_noise = {
        user_id: np.random.normal(0, 0.20)
        for user_id in df["User_ID"].unique()
    }

    intercept = -0.10
    beta_d_total = -1.20
    beta_info = -0.65
    beta_risk = 1.80
    beta_subj = 1.50
    beta_lit = 0.10
    beta_risk_lit = 0.90
    beta_subj_lit = 1.20
    beta_trust = -1.10

    linear = (
        intercept
        + beta_d_total * df["D_total"]
        + beta_info * df["Info"]
        + beta_risk * df["Risk"]
        + beta_subj * df["Subj"]
        + beta_lit * df["Lit"]
        + beta_risk_lit * (df["Risk"] * df["Lit"])
        + beta_subj_lit * (df["Subj"] * df["Lit"])
        + beta_trust * df["Trust_Norm"]
        + df["User_ID"].map(user_noise)
        + np.random.normal(0, 0.12, size=len(df))
    )

    prob = _sigmoid(linear)
    df["P_human"] = np.random.binomial(1, prob, size=len(df)).astype(float)
    df["Scenario_Type"] = "SYNTHETIC_DEMO_NOT_REAL"

    return df


def run_synthetic_scenario():
    df_syn = build_synthetic_dataset()
    df_syn.to_csv("synthetic_data.csv", index=False)

    df_with_trust = run_analysis(
        df_syn,
        include_trust=True,
        output_file="Synthetic_Statistical_Results_With_Trust.txt"
    )
    df_without_trust = run_analysis(
        df_syn,
        include_trust=False,
        output_file="Synthetic_Statistical_Results_Without_Trust.txt"
    )

    visualize_results(df_with_trust, include_trust=True, file_prefix="Synthetic_WithTrust")
    visualize_results(df_without_trust, include_trust=False, file_prefix="Synthetic_WithoutTrust")

    print("Đã tạo scenario mô phỏng (synthetic) cho mục đích demo phương pháp.")
    print("Lưu ý: KHÔNG dùng synthetic_data.csv để báo cáo kết quả khảo sát thật.")


if __name__ == "__main__":
    run_synthetic_scenario()
