import pandas as pd
import numpy as np
import re
import unicodedata
import os

SURVEY_FILE = "Form nghiên cứu.csv"
METADATA_FILE = "Ac_Results_Final.xlsx"
OUTPUT_FILE = "final_data.csv"

SCENARIO_OPTION_MAP = {
    # Scenario_ID 1: Paint purchase choice has no AI/Human label in the form.
    1: {
        "ai_keywords": [
            "dat ship", "giao hang", "hoa toc", "5 lit", "5l", "ship"
        ],
        "human_keywords": [
            "tu di mua", "chay xe", "2 lon", "2 lo", "mua 2"
        ]
    }
}


def _normalize_text(text):
    if pd.isna(text):
        return ""
    text_lower = str(text).lower().strip()
    text_norm = unicodedata.normalize("NFD", text_lower)
    return "".join(ch for ch in text_norm if unicodedata.category(ch) != "Mn")


def parse_phuman(text, scenario_idx=None):
    if pd.isna(text):
        return None
    text_norm = _normalize_text(text)

    if "loi khuyen ai" in text_norm or "loi khuyen cua ai" in text_norm:
        return 0.0
    if "loi khuyen con nguoi" in text_norm or "loi khuyen cua con nguoi" in text_norm:
        return 1.0
    if re.search(r"\bai\b", text_norm) and "con nguoi" not in text_norm:
        return 0.0
    if re.search(r"\bcon nguoi\b", text_norm):
        return 1.0

    mapping = SCENARIO_OPTION_MAP.get(scenario_idx)
    if mapping:
        if any(keyword in text_norm for keyword in mapping["ai_keywords"]):
            return 0.0
        if any(keyword in text_norm for keyword in mapping["human_keywords"]):
            return 1.0

    return None

def get_scenario_attributes(idx):
    risk = 0.0 if idx < 8 else 1.0
    subj = 0.0 if (0 <= idx <= 3) or (8 <= idx <= 11) else 1.0
    info = 0.0 if (idx % 4) < 2 else 1.0
    return risk, subj, info

def preprocess_data(csv_file):
    ac_dict = {}
    meta_path = METADATA_FILE
    if not os.path.exists(meta_path) and os.path.exists(METADATA_FILE + " - Sheet1.csv"):
        meta_path = METADATA_FILE + " - Sheet1.csv"

    if os.path.exists(meta_path):
        try:
            if meta_path.endswith('.csv'):
                meta_df = pd.read_csv(meta_path)
            else:
                meta_df = pd.read_excel(meta_path)
            for i, row in meta_df.iterrows():
                d_total_val = float(row.get('D_total', 0.5)) if pd.notna(row.get('D_total')) else 0.5
                ac_dict[i] = {
                    'AC_Label': float(row.get('AC_Label', 1.0)) if pd.notna(row.get('AC_Label')) else 1.0,
                    'D_total': round(d_total_val, 6)
                }
        except Exception as e:
            print(f"Lỗi đọc Metadata: {e}")

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Lỗi đọc file khảo sát: {e}")
        return None

    long_data = []
    scenario_cols = df.columns[5:21]
    unmapped_by_scenario = {idx: 0 for idx in range(len(scenario_cols))}

    for user_id, row in df.iterrows():
        # Xử lý Literacy: Chuyển thang 1-5 về thang 0-1 (0, 0.25, 0.5, 0.75, 1)
        lit_text = str(row.iloc[3])
        match = re.search(r'(\d+)', lit_text)
        lit_raw = int(match.group(1)) if match else 3
        lit_norm = (lit_raw - 1) / 4.0

        try:
            trust_raw = float(row.iloc[4])
            trust_norm = max(0.0, min(1.0, (trust_raw - 1) / 8.0))
        except:
            trust_norm = 0.5

        for idx, col_name in enumerate(scenario_cols):
            p_human = parse_phuman(row[col_name], scenario_idx=idx)
            if p_human is None:
                unmapped_by_scenario[idx] += 1
                continue

            risk, subj, info = get_scenario_attributes(idx)
            ac_info = ac_dict.get(idx, {'AC_Label': 1.0, 'D_total': 0.5})

            long_data.append({
                'User_ID': user_id,
                'Scenario_ID': idx,
                'AC_Label': ac_info['AC_Label'],
                'D_total': ac_info['D_total'],
                'P_human': p_human,
                'Risk': risk,
                'Subj': subj,
                'Info': info,
                'Lit': lit_norm,       # Đã chuyển về thang [0, 1]
                'Trust_Norm': trust_norm
            })

    clean_df = pd.DataFrame(long_data)
    clean_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Tiền xử lý hoàn tất! Đã lưu: {OUTPUT_FILE}")
    unmapped = {k: v for k, v in unmapped_by_scenario.items() if v}
    if unmapped:
        print("Unmapped responses by Scenario_ID:", unmapped)
    return clean_df