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
    infoload = 0.0 if (idx % 4) < 2 else 1.0
    return risk, subj, infoload


def preprocess_data(csv_file):
    ac_dict = {}
    try:
        if os.path.exists(METADATA_FILE):
            meta_df = pd.read_excel(METADATA_FILE)
        else:
            meta_path = METADATA_FILE + " - Sheet1.csv"
            try:
                meta_df = pd.read_csv(meta_path, encoding='utf-8-sig')
            except:
                meta_df = pd.read_csv(meta_path, encoding='latin1')

        for _, row in meta_df.iterrows():
            idx = int(row['ID']) - 1
            ac_dict[idx] = {
                'AC_Label': float(row['AC_Label']) if pd.notna(row['AC_Label']) else 1.0,
                'D_total': float(row['D_total']) if pd.notna(row['D_total']) else 0.5
            }
    except Exception as e:
        print(f"Lỗi đọc Metadata: {e}")

    try:
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
    except Exception as e:
        print(f"Lỗi đọc file khảo sát: {e}")
        return None

    long_data = []
    scenario_cols = df.columns[5:21]

    for user_id, row in df.iterrows():
        # Xử lý Literacy (Lit_Norm -> gọi là Lit)
        lit_text = str(row.iloc[3])
        match = re.search(r'(\d+)', lit_text)
        lit_raw = int(match.group(1)) if match else 3
        lit_norm = (lit_raw - 1) / 4.0

        # Xử lý Trust (Trust_Norm -> đổi tên thành Trust)
        try:
            trust_raw = float(row.iloc[4])
            trust_ai_norm = max(0.0, min(1.0, (trust_raw - 1) / 8.0))
            trust_final = 1.0 - trust_ai_norm  # 0 = Tin AI, 1 = Tin Human
        except:
            trust_final = 0.5

        for idx, col_name in enumerate(scenario_cols):
            p_human = parse_phuman(row[col_name])
            if p_human is None: continue

            risk, subj, infoload = get_scenario_attributes(idx)
            ac_info = ac_dict.get(idx, {'AC_Label': 1.0, 'D_total': 0.5})

            long_data.append({
                'User_ID': user_id,
                'Scenario_ID': idx,
                'Risk': risk,
                'Subj': subj,
                'InfoLoad': infoload,
                'Lit': lit_norm,
                'Trust': trust_final,  # Tên biến mới
                'D_total': ac_info['D_total'],
                'P_human': p_human
            })

    final_df = pd.DataFrame(long_data)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Tiền xử lý hoàn tất! Đã lưu {len(final_df)} bản ghi.")
    return final_df