import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import pearsonr


def run_analysis(df):
    results = []
    results.append("=== BÁO CÁO KIỂM ĐỊNH MÔ HÌNH: TÁC ĐỘNG TRỰC TIẾP & ĐIỀU TIẾT ===\n")

    # =========================================================
    # 1. KIỂM TRA ĐA CỘNG TUYẾN: XUẤT BẢNG CHUẨN KHOA HỌC (CSV)
    # =========================================================
    var_names = ['Ctx', 'Risk', 'Subj', 'Info', 'AILit', 'Trust', 'DV']
    df_clean = df[var_names].dropna()

    # Tính VIF
    X = df_clean[['Ctx', 'Risk', 'Subj', 'Info', 'AILit', 'Trust']]
    X_with_const = sm.add_constant(X)
    vif_dict = {}
    for col in X.columns:
        idx = X_with_const.columns.get_loc(col)
        vif_dict[col] = variance_inflation_factor(X_with_const.values, idx)

    # Tạo bảng dữ liệu xuất ra CSV chuẩn báo cáo
    n = len(var_names)
    table_data = []

    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append("-")  # Đường chéo chính chuẩn khoa học hay để dấu gạch ngang
            elif i > j:
                corr, p = pearsonr(df_clean[var_names[i]], df_clean[var_names[j]])
                stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                row.append(f"{corr:.3f}{stars}")
            else:
                row.append("")  # Tam giác trên bỏ trống

        # Thêm cột VIF vào cuối
        if var_names[i] == 'DV':
            row.append("-")
        else:
            row.append(f"{vif_dict[var_names[i]]:.2f}")

        table_data.append(row)

    cols = [str(x + 1) for x in range(n)] + ['VIF']
    idx_labels = [f"{i + 1}. {name}" for i, name in enumerate(var_names)]

    # Lưu ra file CSV để copy thẳng vào Word/Excel
    corr_df = pd.DataFrame(table_data, index=idx_labels, columns=cols)
    corr_df.to_csv("Correlation_VIF_Academic.csv", encoding="utf-8-sig")

    results.append("Đã xuất Bảng Ma trận tương quan & VIF chuẩn học thuật ra file 'Correlation_VIF_Academic.csv'.")
    results.append("Vui lòng mở file CSV này để copy vào báo cáo Word/Excel của bạn.\n")

    # =========================================================
    # 2. PHƯƠNG TRÌNH: GEE Binomial DỰ BÁO HÀNH VI (DV)
    # =========================================================
    results.append("--- MÔ HÌNH: GEE BINOMIAL DỰ BÁO HÀNH VI (DV) ---")
    formula = "DV ~ Ctx + Info + Risk + Subj + Trust + AILit + Risk:AILit + Subj:AILit"

    try:
        cov_struct = sm.cov_struct.Exchangeable()
        fam_bin = sm.families.Binomial()
        model_gee = smf.gee(formula, groups=df['User_ID'], data=df, family=fam_bin, cov_struct=cov_struct).fit()
        results.append(model_gee.summary().as_text() + "\n")

        b_ctx = model_gee.params.get('Ctx', 0);
        p_ctx = model_gee.pvalues.get('Ctx', 1)
        b_info = model_gee.params.get('Info', 0);
        p_info = model_gee.pvalues.get('Info', 1)
        b_risk = model_gee.params.get('Risk', 0);
        p_risk = model_gee.pvalues.get('Risk', 1)
        b_subj = model_gee.params.get('Subj', 0);
        p_subj = model_gee.pvalues.get('Subj', 1)
        b_trust = model_gee.params.get('Trust', 0);
        p_trust = model_gee.pvalues.get('Trust', 1)
        b_risk_lit = model_gee.params.get('Risk:AILit', 0);
        p_risk_lit = model_gee.pvalues.get('Risk:AILit', 1)
        b_subj_lit = model_gee.params.get('Subj:AILit', 0);
        p_subj_lit = model_gee.pvalues.get('Subj:AILit', 1)

        results.append("--- KẾT LUẬN GIẢ THUYẾT ---")
        res_h1 = "Ủng hộ" if p_ctx < 0.05 and b_ctx < 0 else "Không ủng hộ"
        results.append(f"[H1] Tác động của Bối cảnh (Ctx): β1 = {b_ctx:.3f}, p = {p_ctx:.3f} => {res_h1}")

        res_h2 = "Ủng hộ" if p_info < 0.05 and b_info < 0 else "Không ủng hộ"
        results.append(f"[H2] Tác động của Tải lượng (Info): β2 = {b_info:.3f}, p = {p_info:.3f} => {res_h2}")

        res_h3 = "Ủng hộ" if p_risk < 0.05 and b_risk > 0 else "Ý nghĩa thống kê nhưng ngược chiều" if p_risk < 0.05 else "Không ủng hộ"
        results.append(f"[H3] Tác động của Mức độ rủi ro (Risk): β3 = {b_risk:.3f}, p = {p_risk:.3f} => {res_h3}")

        res_h4 = "Ủng hộ" if p_subj < 0.05 and b_subj > 0 else "Ý nghĩa thống kê nhưng ngược chiều" if p_subj < 0.05 else "Không ủng hộ"
        results.append(f"[H4] Tác động của Tính chủ quan (Subj): β4 = {b_subj:.3f}, p = {p_subj:.3f} => {res_h4}")

        res_h7 = "Ủng hộ" if p_trust < 0.05 and b_trust > 0 else "Không ủng hộ"
        results.append(f"[H7] Tác động của Niềm tin (Trust): β5 = {b_trust:.3f}, p = {p_trust:.3f} => {res_h7}")

        res_h5 = "Ủng hộ" if p_risk_lit < 0.05 and b_risk_lit > 0 else "Không ủng hộ"
        results.append(
            f"[H5] Am hiểu AI điều tiết Rủi ro (Risk x AILit): β7 = {b_risk_lit:.3f}, p = {p_risk_lit:.3f} => {res_h5}")

        res_h6 = "Ủng hộ" if p_subj_lit < 0.05 and b_subj_lit > 0 else "Không ủng hộ"
        results.append(
            f"[H6] Am hiểu AI điều tiết Lĩnh vực (Subj x AILit): β8 = {b_subj_lit:.3f}, p = {p_subj_lit:.3f} => {res_h6}\n")

    except Exception as e:
        results.append(f"Lỗi chạy mô hình GEE: {e}\n")

    final_report = "\n".join(results)
    print(final_report)
    with open("GEE_Results.txt", "w", encoding="utf-8") as f:
        f.write(final_report)

    return df