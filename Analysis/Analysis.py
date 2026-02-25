import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


def run_analysis(df):
    results = []
    results.append("=== BÁO CÁO KIỂM ĐỊNH MÔ HÌNH: TÁC ĐỘNG TRỰC TIẾP & ĐIỀU TIẾT ===\n")
    results.append(
        "Phương pháp: Generalized Estimating Equations (GEE)\n(Sử dụng cấu trúc Exchangeable để kiểm soát phương sai của 16 kịch bản lặp lại trên cùng 1 người dùng)\n")

    # Cấu trúc hiệp phương sai cho dữ liệu nhóm theo User_ID
    cov_struct = sm.cov_struct.Exchangeable()

    # =========================================================
    # PHƯƠNG TRÌNH: GEE Binomial DỰ BÁO HÀNH VI (P_human)
    # =========================================================
    results.append("--- MÔ HÌNH: GEE BINOMIAL DỰ BÁO HÀNH VI CHỌN CON NGƯỜI ---")

    # Công thức: Tác động của Mâu thuẫn, Tải lượng, Rủi ro, Chủ quan, Niềm tin và 2 biến Điều tiết
    formula = "P_human ~ D_total + InfoLoad + Risk + Subj + Trust + Lit + Risk:Lit + Subj:Lit"

    try:
        fam_bin = sm.families.Binomial()
        model_gee = smf.gee(formula, groups=df['User_ID'], data=df, family=fam_bin, cov_struct=cov_struct).fit()
        results.append(model_gee.summary().as_text() + "\n")

        # Lấy hệ số beta và p-value
        b_dtotal = model_gee.params.get('D_total', 0);
        p_dtotal = model_gee.pvalues.get('D_total', 1)
        b_info = model_gee.params.get('InfoLoad', 0);
        p_info = model_gee.pvalues.get('InfoLoad', 1)
        b_risk = model_gee.params.get('Risk', 0);
        p_risk = model_gee.pvalues.get('Risk', 1)
        b_subj = model_gee.params.get('Subj', 0);
        p_subj = model_gee.pvalues.get('Subj', 1)
        b_trust = model_gee.params.get('Trust', 0);
        p_trust = model_gee.pvalues.get('Trust', 1)
        b_risk_lit = model_gee.params.get('Risk:Lit', 0);
        p_risk_lit = model_gee.pvalues.get('Risk:Lit', 1)
        b_subj_lit = model_gee.params.get('Subj:Lit', 0);
        p_subj_lit = model_gee.pvalues.get('Subj:Lit', 1)

        results.append("--- KẾT LUẬN 7 GIẢ THUYẾT CHÍNH THỨC ---")

        # [H1] Bối cảnh lời khuyên (D_total) -> Hành vi (Kỳ vọng: Âm)
        res_h1 = "Ủng hộ" if p_dtotal < 0.05 and b_dtotal < 0 else "Không ủng hộ"
        results.append(f"[H1] Tác động của Mâu thuẫn (D_total): β = {b_dtotal:.3f}, p = {p_dtotal:.3f} => {res_h1}")

        # [H2] Tải lượng thông tin (InfoLoad_Norm) -> Hành vi (Kỳ vọng: Âm)
        res_h2 = "Ủng hộ" if p_info < 0.05 and b_info < 0 else "Không ủng hộ"
        results.append(f"[H2] Tác động của Tải lượng (InfoLoad): β = {b_info:.3f}, p = {p_info:.3f} => {res_h2}")

        # [H3] Mức độ rủi ro (Risk) -> Hành vi (Kỳ vọng: Dương)
        res_h3 = "Ủng hộ" if p_risk < 0.05 and b_risk > 0 else "Ý nghĩa thống kê nhưng ngược chiều" if p_risk < 0.05 else "Không ủng hộ"
        results.append(f"[H3] Tác động của Mức độ rủi ro (Risk): β = {b_risk:.3f}, p = {p_risk:.3f} => {res_h3}")

        # [H4] Lĩnh vực câu hỏi (Subj) -> Hành vi (Kỳ vọng: Dương)
        res_h4 = "Ủng hộ" if p_subj < 0.05 and b_subj > 0 else "Ý nghĩa thống kê nhưng ngược chiều" if p_subj < 0.05 else "Không ủng hộ"
        results.append(f"[H4] Tác động của Tính chủ quan (Subj): β = {b_subj:.3f}, p = {p_subj:.3f} => {res_h4}")

        # [H5] Điều tiết của Am hiểu AI (Risk x Lit) -> Hành vi (Kỳ vọng: Dương)
        res_h5 = "Ủng hộ" if p_risk_lit < 0.05 and b_risk_lit > 0 else "Không ủng hộ"
        results.append(
            f"[H5] Am hiểu AI điều tiết Rủi ro (Risk x Lit): β = {b_risk_lit:.3f}, p = {p_risk_lit:.3f} => {res_h5}")

        # [H6] Điều tiết của Am hiểu AI (Subj x Lit) -> Hành vi (Kỳ vọng: Dương)
        res_h6 = "Ủng hộ" if p_subj_lit < 0.05 and b_subj_lit > 0 else "Không ủng hộ"
        results.append(
            f"[H6] Am hiểu AI điều tiết Lĩnh vực (Subj x Lit): β = {b_subj_lit:.3f}, p = {p_subj_lit:.3f} => {res_h6}")

        # [H7] Mức độ tin cậy (Trust) -> Hành vi (Kỳ vọng: Dương)
        res_h7 = "Ủng hộ" if p_trust < 0.05 and b_trust > 0 else "Không ủng hộ"
        results.append(f"[H7] Tác động của Niềm tin (Trust): β = {b_trust:.3f}, p = {p_trust:.3f} => {res_h7}\n")

    except Exception as e:
        results.append(f"Lỗi chạy mô hình GEE: {e}\n")

    final_report = "\n".join(results)
    print(final_report)
    with open("GEE_Results.txt", "w", encoding="utf-8") as f:
        f.write(final_report)

    return df