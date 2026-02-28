import pandas as pd
import numpy as np
import re
import os

SURVEY_FILE = "Form nghiên cứu.csv"
METADATA_FILE = "Ac_Results_Final.xlsx"
OUTPUT_FILE = "final_data.csv"

def parse_dv(text):
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
    try:
        if os.path.exists(METADATA_FILE):
            meta_df = pd.read_excel(METADATA_FILE)
        else:
            meta_path = METADATA_FILE + " - Sheet1.csv"
            meta_df = pd.read_csv(meta_path, encoding='utf-8-sig')

        for _, row in meta_df.iterrows():
            idx = int(row['ID']) - 1
            ac_dict[idx] = {
                'Ctx': float(row['D_total']) if pd.notna(row['D_total']) else 0.5  # Đổi tên thành Ctx
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
        # Xử lý Literacy (Lit_Norm -> gọi là AILit)
        lit_text = str(row.iloc[3])
        match = re.search(r'(\d+)', lit_text)
        lit_raw = int(match.group(1)) if match else 3
        ailit_norm = (lit_raw - 1) / 4.0

        # Xử lý Trust
        try:
            trust_raw = float(row.iloc[4])
            trust_ai_norm = max(0.0, min(1.0, (trust_raw - 1) / 8.0))
            trust_final = 1.0 - trust_ai_norm  # 0 = Tin AI, 1 = Tin Human
        except:
            trust_final = 0.5

        for idx, col_name in enumerate(scenario_cols):
            dv_val = parse_dv(row[col_name])
            if dv_val is None: continue

            risk, subj, info = get_scenario_attributes(idx)
            ac_info = ac_dict.get(idx, {'Ctx': 0.5})

            long_data.append({
                'User_ID': user_id,
                'Scenario_ID': idx,
                'Ctx': ac_info['Ctx'],     # Bối cảnh lời khuyên
                'Risk': risk,              # Rủi ro
                'Subj': subj,              # Tính chủ quan
                'Info': info,              # Tải lượng thông tin
                'AILit': ailit_norm,       # Am hiểu AI
                'Trust': trust_final,      # Niềm tin
                'DV': dv_val               # Hành vi chấp nhận (P_human cũ)
            })

    final_df = pd.DataFrame(long_data)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Tiền xử lý hoàn tất! Đã lưu {len(final_df)} bản ghi.")
    return final_df