import pandas as pd
import numpy as np
import re
import os

SURVEY_FILE = "Form nghiên cứu.csv"
METADATA_FILE = "Ac_Results_Final.xlsx"
OUTPUT_FILE = "final_data.csv"

def parse_phuman(text):
    if pd.isna(text): return None
    text_lower = str(text).lower()
    if 'lời khuyên ai' in text_lower or 'lời khuyên của ai' in text_lower: return 0.0
    if 'lời khuyên con người' in text_lower or 'lời khuyên của con người' in text_lower: return 1.0
    if 'ai' in text_lower and 'con người' not in text_lower: return 0.0
    if 'con người' in text_lower: return 1.0
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

    for user_id, row in df.iterrows():
        # Xử lý Literacy giữ nguyên thang 1-5

        lit_text = str(row.iloc[3])
        match = re.search(r'(\d+)', lit_text)
        lit_raw = int(match.group(1)) if match else 3

        try:
            trust_raw = float(row.iloc[4])
            trust_norm = max(0.0, min(1.0, (trust_raw - 1) / 4.0))
        except:
            trust_norm = 0.5

        for idx, col_name in enumerate(scenario_cols):
            p_human = parse_phuman(row[col_name])
            if p_human is None: continue

            risk, subj, info = get_scenario_attributes(idx)
            ac_info = ac_dict.get(idx, {'AC_Label': 1.0, 'D_total': 0.5})

            long_data.append({
                'User_ID': user_id,
                'Scenario_ID': idx,
                'AC_Label': ac_info['AC_Label'],
                'D_total': ac_info['D_total'],
                'P_human': p_human,      # ĐÃ ĐỔI TÊN
                'Risk': risk,
                'Subj': subj,
                'Info': info,
                'Lit': lit_raw,          # ĐÃ GIỮ NGUYÊN 1-5
                'Trust_Norm': trust_norm
            })

    clean_df = pd.DataFrame(long_data)
    clean_df.to_csv(OUTPUT_FILE, index=False)
    print(f"done")
    return clean_df