import statsmodels.api as sm
import statsmodels.formula.api as smf


def _fit_binary_model(formula, data):
    try:
        return smf.logit(formula, data=data).fit(disp=0), "Logit"
    except Exception:
        glm = smf.glm(formula=formula, data=data, family=sm.families.Binomial()).fit()
        return glm, "GLM-Binomial"


def _fit_binary_model_clustered(formula, data, group_col):
    gee = smf.gee(
        formula=formula,
        groups=group_col,
        data=data,
        family=sm.families.Binomial()
    ).fit()
    return gee, "GEE-Binomial (cluster theo User_ID)"


def _judge_hypothesis(beta, p_value, expected_sign):
    if p_value >= 0.05:
        return "Không ủng hộ"

    if expected_sign == "positive" and beta > 0:
        return "Ủng hộ"
    if expected_sign == "negative" and beta < 0:
        return "Ủng hộ"

    return "Không ủng hộ (có ý nghĩa nhưng ngược chiều giả thuyết)"


def _has_variation(series):
    return series.nunique(dropna=True) > 1


def run_analysis(df, include_trust=True, output_file="Statistical_Results.txt"):
    case_name = "CÓ TRUST" if include_trust else "KHÔNG TRUST"
    results = []
    results.append(f"=== BÁO CÁO KIỂM ĐỊNH MÔ HÌNH HIỆN TẠI (H1-H7) - {case_name} ===\n")

    required_cols = ['P_human', 'D_total', 'Risk', 'Subj', 'Info', 'Lit']
    if include_trust:
        required_cols.append('Trust_Norm')

    model_df = df.dropna(subset=required_cols).copy()
    results.append(f"Số quan sát đưa vào mô hình: N = {len(model_df)}")
    if 'User_ID' in model_df.columns:
        results.append(f"Số người trả lời: {model_df['User_ID'].nunique()}")

    if 'AC_Label' in model_df.columns and not _has_variation(model_df['AC_Label']):
        results.append("Cảnh báo: AC_Label không có biến thiên (toàn bộ cùng 1 giá trị), không thể kiểm định theo biến nhị phân Consensus/Conflict.")

    if not _has_variation(model_df['D_total']):
        results.append("Cảnh báo: D_total gần như hằng số, H1 sẽ rất khó có ý nghĩa thống kê.")

    formula = "P_human ~ D_total + Info + Risk + Subj + Lit + Risk:Lit + Subj:Lit"
    if include_trust:
        formula += " + Trust_Norm"

    results.append(f"Công thức mô hình: {formula}\n")

    try:
        if 'User_ID' in model_df.columns:
            fitted_model, model_type = _fit_binary_model_clustered(formula, model_df, 'User_ID')
        else:
            fitted_model, model_type = _fit_binary_model(formula, model_df)
        results.append(f"Loại mô hình: {model_type}")
        results.append(fitted_model.summary().as_text() + "\n")

        params = fitted_model.params
        pvalues = fitted_model.pvalues

        checks = [
            ("H1", "D_total", "negative", "Bối cảnh lời khuyên -> Chọn AI hơn"),
            ("H2", "Info", "negative", "Thông tin chi tiết -> Chọn AI hơn (chuyên gia)"),
            ("H3", "Risk", "negative", "Rủi ro cao -> Chọn AI hơn (chuyên gia)"),
            ("H4", "Subj", "negative", "Chủ đề chủ quan -> Chọn AI hơn (dựa vào training)"),
            ("H5", "Risk:Lit", "positive", "Am hiểu AI điều tiết Risk -> Hành vi"),
            ("H6", "Subj:Lit", "positive", "Am hiểu AI điều tiết Subj -> Hành vi"),
        ]

        if include_trust:
            checks.append(("H7", "Trust_Norm", "negative", "Mức độ tin cậy vào AI -> Hành vi chọn Con người"))

        results.append("--- KẾT LUẬN GIẢ THUYẾT ---")
        for hypothesis, term, sign, label in checks:
            beta = params.get(term, 0.0)
            p_val = pvalues.get(term, 1.0)
            conclusion = _judge_hypothesis(beta, p_val, sign)
            results.append(
                f"[{hypothesis}] {label}: β = {beta:.4f}, p = {p_val:.4f} => {conclusion}"
            )

        if include_trust:
            results.append("Ghi chú H7: vì DV là P_human (1 = chọn Con người), nếu Trust_Norm đo niềm tin vào AI thì dấu kỳ vọng hợp lý là âm.")
        else:
            results.append("[H7] Không kiểm định trong chế độ KHÔNG TRUST.")

    except Exception as error:
        results.append(f"Lỗi khi chạy mô hình: {error}")

    output_text = "\n".join(results)
    print(output_text)

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(output_text)

    return model_df