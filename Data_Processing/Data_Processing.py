import pandas as pd
import numpy as np
import re


def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    # Mapping 20 tình huống dựa trên bảng định lượng D_total
    scenario_meta = [
        {'idx': 5, 'risk': 0.0, 'subj': 0, 'd_total': 0.222},  # 1. Xếp cốp xe
        {'idx': 6, 'risk': 0.0, 'subj': 0, 'd_total': 0.236},  # 2. Tiệc BBQ
        {'idx': 7, 'risk': 0.0, 'subj': 0, 'd_total': 0.111},  # 3. Xếp kệ sách
        {'idx': 8, 'risk': 0.0, 'subj': 0, 'd_total': 0.561},  # 4. Game
        {'idx': 9, 'risk': 0.0, 'subj': 1, 'd_total': 0.889},  # 5. Caption ảnh
        {'idx': 10, 'risk': 0.0, 'subj': 1, 'd_total': 0.333},  # 6. Chọn phim
        {'idx': 11, 'risk': 0.0, 'subj': 1, 'd_total': 0.778},  # 7. Cây để bàn
        {'idx': 12, 'risk': 0.5, 'subj': 0, 'd_total': 0.222},  # 8. Máy tính chậm
        {'idx': 13, 'risk': 0.5, 'subj': 0, 'd_total': 0.222},  # 9. Ảnh hồ sơ
        {'idx': 14, 'risk': 0.5, 'subj': 0, 'd_total': 0.111},  # 10. Chanh nóng
        {'idx': 15, 'risk': 0.5, 'subj': 0, 'd_total': 0.222},  # 11. Sửa vòi nước
        {'idx': 16, 'risk': 0.5, 'subj': 1, 'd_total': 0.889},  # 12. Từ chối đám cưới
        {'idx': 17, 'risk': 0.5, 'subj': 1, 'd_total': 1.000},  # 13. Hàng xóm ồn
        {'idx': 18, 'risk': 0.5, 'subj': 1, 'd_total': 0.111},  # 14. Nhiệm vụ quá tải
        {'idx': 19, 'risk': 1.0, 'subj': 0, 'd_total': 0.778},  # 15. Thưởng Tết
        {'idx': 20, 'risk': 1.0, 'subj': 0, 'd_total': 0.778},  # 16. Mua nhà
        {'idx': 21, 'risk': 1.0, 'subj': 0, 'd_total': 0.889},  # 17. Cắt lỗ cổ phiếu
        {'idx': 22, 'risk': 1.0, 'subj': 1, 'd_total': 1.000},  # 18. Sự nghiệp/Gia đình
        {'idx': 23, 'risk': 1.0, 'subj': 1, 'd_total': 0.111},  # 19. Bố mẹ già
        {'idx': 24, 'risk': 1.0, 'subj': 1, 'd_total': 0.111},  # 20. Burnout
    ]

    long_data = []
    for row_idx, row in df.iterrows():
        # Xử lý Literacy (1-5) và Trust (1-9 -> 0-1)
        lit_match = re.search(r'Mức (\d+)', str(row.iloc[3]))
        literacy = int(lit_match.group(1)) if lit_match else 3
        trust_base = (float(row.iloc[4]) - 1) / 8.0 if not pd.isna(row.iloc[4]) else 0.5

        # Age & Gender
        age_str = str(row.iloc[1])
        age = 1 if '18' in age_str else (2 if '23' in age_str else (3 if '31' in age_str else 4))
        gender = 1 if 'nam' in str(row.iloc[2]).lower() else 0

        for i, meta in enumerate(scenario_meta):
            choice = str(row.iloc[5 + i]).lower()
            woa = 1.0 if 'con người' in choice or 'human' in choice else 0.0

            long_data.append({
                'Age': age, 'Gender': gender, 'Literacy': literacy, 'Trust_Base': trust_base,
                'Risk': meta['risk'], 'Subj': meta['subj'], 'D_Total': meta['d_total'], 'WOA': woa,
                'Risk_Label': 'Thấp' if meta['risk'] == 0 else ('Vừa' if meta['risk'] == 0.5 else 'Cao'),
                'Subj_Label': 'Khách quan' if meta['subj'] == 0 else 'Chủ quan'
            })
    return pd.DataFrame(long_data)