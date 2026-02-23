import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

def run_analysis(df):
    results = []
    results.append("=== BÁO CÁO KIỂM ĐỊNH CHI TIẾT 10 GIẢ THUYẾT (H1 - H10) ===\n")

    # =========================================================
    # KIỂM ĐỊNH H1 (Dùng toàn bộ dữ liệu có D_total)
    # =========================================================
    df_h1 = df.dropna(subset=['P_human', 'D_total'])
    results.append("[H1] Tác động của Mâu thuẫn (D_total) đến Hành vi:")
    try:
        model_h1 = smf.logit("P_human ~ D_total", data=df_h1).fit(disp=0)
        results.append(model_h1.summary().as_text() + "\n")

        b_h1 = model_h1.params.get('D_total', 0)
        p_h1 = model_h1.pvalues.get('D_total', 1)

        # Mâu thuẫn có tác động ngược chiều đến Hành vi chấp nhận lời khuyên
        res_h1 = "Ủng hộ" if p_h1 < 0.1 else "Không ủng hộ"
        results.append(f"=> Kết luận H1: Hệ số β = {b_h1:.3f}, p = {p_h1:.3f} -> {res_h1}.\n")
    except Exception as e:
        results.append(f"=> Lỗi kiểm định H1: {e}\n")

    # =========================================================
    # LỌC TẬP MÂU THUẪN (AC_Label == 1) CHO CÁC GIẢ THUYẾT KHÁC
    # =========================================================
    df_conflict = df[df['AC_Label'] == 1.0].dropna(subset=['P_human'])
    results.append(f"--- PHÂN TÍCH TRÊN TẬP MÂU THUẪN (N={len(df_conflict)}) ---\n")

    p_subj_trust = 1.0
    p_trust_phuman = 1.0

    # 1. MÔ HÌNH OLS DỰ BÁO NIỀM TIN (H4, H5, H7, H10)
    results.append("1. MÔ HÌNH OLS (DV: Niềm tin - Trust_Norm)")
    formula_trust = "Trust_Norm ~ Risk + Subj + Lit + Risk:Lit + Subj:Lit + Risk:Subj"
    try:
        model_trust = smf.ols(formula_trust, data=df_conflict).fit()
        results.append(model_trust.summary().as_text() + "\n")

        t_b = model_trust.params
        t_p = model_trust.pvalues

        # H4: Subj -> Trust (Kỳ vọng: Âm)
        b_subj = t_b.get('Subj', 0)
        p_subj = t_p.get('Subj', 1)
        p_subj_trust = p_subj
        res_h4 = "Ủng hộ" if (p_subj < 0.05 and b_subj < 0) else "Không ủng hộ"
        results.append(f"- [H4] Tính chủ quan -> Niềm tin: α2 = {b_subj:.3f}, p = {p_subj:.3f} => {res_h4}")

        # H5: Lit điều tiết Risk -> Trust
        b_rl = t_b.get('Risk:Lit', 0)
        p_rl = t_p.get('Risk:Lit', 1)
        res_h5 = "Ủng hộ" if p_rl < 0.05 else "Không ủng hộ"
        results.append(f"- [H5] Rủi ro x Am hiểu AI -> Niềm tin: α4 = {b_rl:.3f}, p = {p_rl:.3f} => {res_h5}")

        # H7: Lit điều tiết Subj -> Trust (Kỳ vọng: Âm)
        b_sl = t_b.get('Subj:Lit', 0)
        p_sl = t_p.get('Subj:Lit', 1)
        res_h7 = "Ủng hộ" if (p_sl < 0.05 and b_sl < 0) else "Không ủng hộ"
        results.append(f"- [H7] Chủ quan x Am hiểu AI -> Niềm tin: α5 = {b_sl:.3f}, p = {p_sl:.3f} => {res_h7}")

        # H10: Risk điều tiết Subj -> Trust (Kỳ vọng: Âm)
        b_rs_t = t_b.get('Risk:Subj', 0)
        p_rs_t = t_p.get('Risk:Subj', 1)
        res_h10 = "Ủng hộ" if (p_rs_t < 0.05 and b_rs_t < 0) else "Không ủng hộ"
        results.append(f"- [H10] Rủi ro x Chủ quan -> Niềm tin: α6 = {b_rs_t:.3f}, p = {p_rs_t:.3f} => {res_h10}\n")

    except Exception as e:
        results.append(f"Lỗi mô hình OLS (Niềm tin): {e}\n")

    # 2. MÔ HÌNH LOGIT DỰ BÁO HÀNH VI (H2, H3, H6, H8, H9)
    results.append("2. MÔ HÌNH LOGIT (DV: P_human - Chọn Con người)")
    # Loại bỏ Risk:Lit khỏi phương trình Logit theo đúng công thức
    formula_phuman = "P_human ~ Risk + Subj + Info + Lit + Trust_Norm + Risk:Subj + Risk:Info"
    try:
        model_phuman = smf.logit(formula_phuman, data=df_conflict).fit(disp=0)
        results.append(model_phuman.summary().as_text() + "\n")

        w_b = model_phuman.params
        w_p = model_phuman.pvalues

        # H2: Trust -> P_human (Kỳ vọng: Âm, vì tin AI thì giảm chọn người)
        b_trust = w_b.get('Trust_Norm', 0)
        p_trust = w_p.get('Trust_Norm', 1)
        p_trust_phuman = p_trust
        res_h2 = "Ủng hộ" if (p_trust < 0.05 and b_trust < 0) else "Không ủng hộ"
        results.append(f"- [H2] Niềm tin -> Hành vi: β5 = {b_trust:.3f}, p = {p_trust:.3f} => {res_h2}")

        # H3: Risk x Subj -> P_human (Kỳ vọng: Dương)
        b_rs_w = w_b.get('Risk:Subj', 0)
        p_rs_w = w_p.get('Risk:Subj', 1)
        res_h3 = "Ủng hộ" if (p_rs_w < 0.05 and b_rs_w > 0) else "Không ủng hộ"
        results.append(f"- [H3] Rủi ro x Chủ quan -> Hành vi: β6 = {b_rs_w:.3f}, p = {p_rs_w:.3f} => {res_h3}")

        # H6: Info -> P_human (Kỳ vọng: Âm)
        b_info = w_b.get('Info', 0)
        p_info = w_p.get('Info', 1)
        res_h6 = "Ủng hộ" if (p_info < 0.05 and b_info < 0) else "Không ủng hộ"
        results.append(f"- [H6] Tải lượng thông tin -> Hành vi: β3 = {b_info:.3f}, p = {p_info:.3f} => {res_h6}")

        # H8: Risk x Info -> P_human (Kỳ vọng: Dương)
        b_ri = w_b.get('Risk:Info', 0)
        p_ri = w_p.get('Risk:Info', 1)
        if p_ri < 0.05 and b_ri > 0:
            res_h8 = "Ủng hộ"
        elif p_ri < 0.05 and b_ri < 0:
            res_h8 = "Không ủng hộ (có ý nghĩa nhưng ngược chiều dự đoán)"
        else:
            res_h8 = "Không ủng hộ"
        results.append(f"- [H8] Rủi ro x Thông tin -> Hành vi: β7 = {b_ri:.3f}, p = {p_ri:.3f} => {res_h8}")

        # H9: Trust là biến trung gian (Causal Step Approach)
        res_h9 = "Ủng hộ" if (p_subj_trust < 0.05 and p_trust_phuman < 0.05) else "Không ủng hộ"
        results.append(f"- [H9] Biến trung gian (Niềm tin): Chuỗi (Subj -> Trust -> P_human) => {res_h9}\n")

    except Exception as e:
        results.append(f"Lỗi mô hình Logit (Hành vi): {e}\n")

    output_text = "\n".join(results)
    print(output_text)

    with open("Statistical_Results.txt", "w", encoding="utf-8") as f:
        f.write(output_text)

    return df_conflict