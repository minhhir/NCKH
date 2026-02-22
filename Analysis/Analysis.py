import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def run_analysis(df):
    results = []
    results.append("=== BÁO CÁO KIỂM ĐỊNH CHI TIẾT 10 GIẢ THUYẾT (H1 - H10) ===\n")

    # =========================================================
    # KIỂM ĐỊNH H1 (Dùng toàn bộ dữ liệu có D_total)
    # =========================================================
    df_h1 = df.dropna(subset=['WOA', 'D_total'])
    results.append(f"[H1] Tác động của Mâu thuẫn (D_total) đến Hành vi:")
    try:
        model_h1 = smf.logit("WOA ~ D_total", data=df_h1).fit(disp=0)
        b_h1 = model_h1.params.get('D_total', 0)
        p_h1 = model_h1.pvalues.get('D_total', 1)
        res_h1 = "ỦNG HỘ" if p_h1 < 0.05 else "KHÔNG ỦNG HỘ"
        results.append(f"- Hệ số D_total: β = {b_h1:.3f}, p = {p_h1:.3f}")
        results.append(f"=> Kết luận H1: {res_h1}\n")
    except Exception as e:
        results.append(f"[H1] Lỗi: {e}\n")

    # =========================================================
    # LỌC TẬP MÂU THUẪN (AC_Label == 1) CHO CÁC GIẢ THUYẾT KHÁC
    # =========================================================
    df_conflict = df[df['AC_Label'] == 1.0].dropna(subset=['WOA'])
    results.append(f"--- PHÂN TÍCH TRÊN TẬP MÂU THUẪN (N={len(df_conflict)}) ---\n")

    # =========================================================
    # MÔ HÌNH 1: OLS DỰ BÁO NIỀM TIN (H4, H5, H7, H10)
    # =========================================================
    formula_trust = "Trust_Norm ~ Risk + Subj + Lit_Norm + Risk:Lit_Norm + Subj:Lit_Norm + Risk:Subj"
    try:
        model_trust = smf.ols(formula_trust, data=df_conflict).fit()
        results.append("1. MÔ HÌNH OLS (DV: Niềm tin - Trust_Norm)")
        results.append(model_trust.summary().as_text() + "\n")

        t_b = model_trust.params
        t_p = model_trust.pvalues

        # H4a & H4b
        b_subj = t_b.get('Subj', 0);
        p_subj = t_p.get('Subj', 1)
        results.append(f"[H4a, H4b] Lĩnh vực tác động lên Niềm tin:")
        results.append(f"- β_Subj = {b_subj:.3f}, p = {p_subj:.3f}")
        if p_subj < 0.05:
            if b_subj < 0:
                results.append("=> ỦNG HỘ H4a & H4b: Nhiệm vụ chủ quan (Subj=1) làm GIẢM niềm tin vào AI.\n")
            else:
                results.append("=> Bác bỏ chiều hướng: Nhiệm vụ chủ quan làm TĂNG niềm tin vào AI.\n")
        else:
            results.append("=> KHÔNG ỦNG HỘ H4a & H4b (p > 0.05).\n")

        # H5a & H5b (Tác động lên Niềm tin)
        b_rl = t_b.get('Risk:Lit_Norm', 0);
        p_rl = t_p.get('Risk:Lit_Norm', 1)
        results.append(f"[H5a, H5b] Am hiểu AI điều tiết Rủi ro lên Niềm tin (Risk x Lit):")
        results.append(f"- β_Risk:Lit = {b_rl:.3f}, p = {p_rl:.3f}")
        if p_rl < 0.05:
            results.append("=> ỦNG HỘ H5a & H5b: Am hiểu AI bẻ lái cách tin tưởng khi gặp rủi ro.\n")
        else:
            results.append("=> KHÔNG ỦNG HỘ H5a & H5b (trên khía cạnh Niềm tin).\n")

        # H7
        b_sl = t_b.get('Subj:Lit_Norm', 0);
        p_sl = t_p.get('Subj:Lit_Norm', 1)
        results.append(f"[H7] Am hiểu AI điều tiết Tính chủ quan lên Niềm tin (Subj x Lit):")
        results.append(f"- β_Subj:Lit = {b_sl:.3f}, p = {p_sl:.3f}")
        if p_sl < 0.05 and b_sl < 0:
            results.append("=> ỦNG HỘ H7: Càng am hiểu AI càng ít tin AI trong tác vụ Chủ quan.\n")
        else:
            results.append("=> KHÔNG ỦNG HỘ H7.\n")

        # H10
        b_rs_t = t_b.get('Risk:Subj', 0);
        p_rs_t = t_p.get('Risk:Subj', 1)
        results.append(f"[H10] Rủi ro khuếch đại tác động Chủ quan lên Niềm tin (Risk x Subj):")
        results.append(f"- β_Risk:Subj = {b_rs_t:.3f}, p = {p_rs_t:.3f}")
        if p_rs_t < 0.05 and b_rs_t < 0:
            results.append("=> ỦNG HỘ H10: Rủi ro cao + Chủ quan -> Niềm tin AI chạm đáy.\n")
        else:
            results.append("=> KHÔNG ỦNG HỘ H10.\n")

    except Exception as e:
        results.append(f"Lỗi mô hình OLS: {e}\n")

    # =========================================================
    # MÔ HÌNH 2: LOGIT DỰ BÁO HÀNH VI (H2, H3, H5b mở rộng, H6, H8)
    # =========================================================
    # Bổ sung Risk:Lit_Norm để kiểm định hiệu ứng "trú ẩn an toàn" lên hành vi thực tế
    formula_woa = "WOA ~ Risk + Subj + Info + Lit_Norm + Trust_Norm + Risk:Lit_Norm + Risk:Subj + Risk:Info"
    try:
        model_woa = smf.logit(formula_woa, data=df_conflict).fit(disp=0)
        results.append("2. MÔ HÌNH LOGIT (DV: WOA - Chọn Con người)")
        results.append(model_woa.summary().as_text() + "\n")

        w_b = model_woa.params
        w_p = model_woa.pvalues

        # H2
        b_trust = w_b.get('Trust_Norm', 0);
        p_trust = w_p.get('Trust_Norm', 1)
        results.append(f"[H2] Cơ chế Niềm tin (Trust) quyết định Hành vi:")
        results.append(f"- β_Trust = {b_trust:.3f}, p = {p_trust:.3f}")
        if p_trust < 0.05 and b_trust < 0:
            results.append("=> ỦNG HỘ H2: Tin AI tăng -> Tỷ lệ chọn Con người giảm.\n")
        else:
            results.append("=> KHÔNG ỦNG HỘ H2.\n")

        # H3
        b_rs_w = w_b.get('Risk:Subj', 0);
        p_rs_w = w_p.get('Risk:Subj', 1)
        results.append(f"[H3] Rủi ro khuếch đại mâu thuẫn Chủ quan (Risk x Subj):")
        results.append(f"- β_Risk:Subj = {b_rs_w:.3f}, p = {p_rs_w:.3f}")
        if p_rs_w < 0.05 and b_rs_w > 0:
            results.append("=> ỦNG HỘ H3: Rủi ro cao + Chủ quan -> Càng bỏ chạy về phía Con người.\n")
        else:
            results.append("=> KHÔNG ỦNG HỘ H3.\n")

        # H5b (Mở rộng trên hành vi)
        b_rl_w = w_b.get('Risk:Lit_Norm', 0);
        p_rl_w = w_p.get('Risk:Lit_Norm', 1)
        results.append(f"[H5 bổ sung] Am hiểu AI điều tiết Rủi ro lên Hành vi (Risk x Lit):")
        results.append(f"- β_Risk:Lit = {b_rl_w:.3f}, p = {p_rl_w:.3f}")
        if p_rl_w < 0.05:
            results.append("=> ỦNG HỘ: Kiến thức AI trực tiếp thay đổi hành vi chọn lựa khi gặp rủi ro.\n")
        else:
            results.append("=> Tương tác này chỉ ảnh hưởng đến Niềm tin, không tác động trực tiếp lên Hành vi.\n")

        # H6
        b_info = w_b.get('Info', 0);
        p_info = w_p.get('Info', 1)
        results.append(f"[H6] Tải lượng thông tin (Info Load) tác động lên Hành vi:")
        results.append(f"- β_Info = {b_info:.3f}, p = {p_info:.3f}")
        if p_info < 0.05 and b_info < 0:
            results.append("=> ỦNG HỘ H6: Quá tải thông tin -> Giao phó cho AI (Giảm chọn người).\n")
        else:
            results.append("=> KHÔNG ỦNG HỘ H6.\n")

        # H8
        b_ri = w_b.get('Risk:Info', 0);
        p_ri = w_p.get('Risk:Info', 1)
        results.append(f"[H8] Tải lượng thông tin suy yếu Rủi ro (Risk x Info):")
        results.append(f"- β_Risk:Info = {b_ri:.3f}, p = {p_ri:.3f}")
        if p_ri < 0.05 and b_ri < 0:
            results.append("=> ỦNG HỘ H8: Info cao khiến người dùng bất chấp rủi ro, vẫn chọn AI.\n")
        else:
            results.append("=> KHÔNG ỦNG HỘ H8.\n")

        # =========================================================
        # KIỂM ĐỊNH H9: MEDIATION TỪNG BƯỚC (Baron & Kenny)
        # =========================================================
        results.append(f"[H9] Niềm tin là biến trung gian (Subj -> Trust -> WOA):")
        try:
            # Chạy mô hình phụ không có Trust để lấy tác động tổng (Total Effect)
            model_woa_no_trust = smf.logit("WOA ~ Risk + Subj + Info + Lit_Norm", data=df_conflict).fit(disp=0)
            p_subj_total = model_woa_no_trust.pvalues.get('Subj', 1)

            results.append(f"- Bước 1 (IV -> DV): Tác động tổng của Subj lên WOA (p = {p_subj_total:.3f})")
            results.append(f"- Bước 2 (IV -> Med): Tác động của Subj lên Trust (p = {p_subj:.3f})")
            results.append(f"- Bước 3 (Med -> DV): Tác động của Trust lên WOA (p = {p_trust:.3f})")

            if p_subj < 0.05 and p_trust < 0.05:
                results.append("=> ỦNG HỘ H9: Chuỗi trung gian Subj -> Trust -> WOA hợp lệ.\n")
            else:
                results.append("=> KHÔNG ỦNG HỘ H9: Chuỗi trung gian bị đứt gãy.\n")
        except Exception as e:
            results.append(f"Lỗi kiểm định H9: {e}\n")

    except Exception as e:
        results.append(f"Lỗi mô hình Logit: {e}\n")

    with open("Statistical_Results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(results))
    print("-> Đã hoàn thành 10 giả thuyết. Kết quả lưu tại 'Statistical_Results.txt'")

    return df_conflict