import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def visualize_results(df):
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # Đặt label cho các biến để biểu đồ trực quan hơn
    df['Risk_Label'] = df['Risk'].map({0.0: 'Rủi ro Thấp', 1.0: 'Rủi ro Cao'})
    df['Subj_Label'] = df['Subj'].map({0.0: 'Khách quan', 1.0: 'Chủ quan'})
    # SỬA LỖI Ở ĐÂY: Dùng cột 'Info' để tạo 'Info_Label' thay vì InfoLoad_Norm
    df['Info_Label'] = df['InfoLoad'].map({0.0: 'Tải thấp', 1.0: 'Tải cao'})
    df['Mức độ Am hiểu AI (Lit)'] = df['Lit'].apply(lambda x: 'Am hiểu Cao' if x >= 0.75 else 'Am hiểu Thấp/TB')

    # Chia nhóm D_total (Mâu thuẫn) thành 2 mức để vẽ barplot
    median_d = df['D_total'].median()
    df['Conflict_Level'] = df['D_total'].apply(lambda x: 'Mâu thuẫn Cao' if x > median_d else 'Mâu thuẫn Thấp')

    # CHART 1: H1 (Mâu thuẫn -> Hành vi)
    plt.figure(figsize=(6, 5))
    ax1 = sns.barplot(data=df, x='Conflict_Level', y='P_human', hue='Conflict_Level', errorbar=None, palette='pastel',
                      legend=False)
    plt.title('[H1] Tác động của Mâu thuẫn đến Hành vi', fontweight='bold')
    plt.ylabel('Tỷ lệ chọn Con người (P_human)')
    plt.ylim(0, 1.2)
    for container in ax1.containers: ax1.bar_label(container, fmt='%.2f', padding=3)
    plt.tight_layout()
    plt.savefig('Chart_01_H1_Dtotal_Phuman.png', dpi=300)
    plt.close()

    # CHART 2: H2 (Tải lượng thông tin -> Hành vi)
    plt.figure(figsize=(6, 5))
    # SỬA LỖI Ở ĐÂY: Trục x và hue dùng 'Info_Label'
    ax2 = sns.barplot(data=df, x='Info_Label', y='P_human', hue='Info_Label', errorbar=None, palette='Set2',
                      legend=False)
    plt.title('[H2] Tác động của Tải lượng thông tin đến Hành vi', fontweight='bold')
    plt.ylabel('Tỷ lệ chọn Con người (P_human)')
    plt.ylim(0, 1.2)
    for container in ax2.containers: ax2.bar_label(container, fmt='%.2f', padding=3)
    plt.tight_layout()
    plt.savefig('Chart_02_H2_Info_Phuman.png', dpi=300)
    plt.close()

    # CHART 3: H3 (Rủi ro -> Hành vi)
    plt.figure(figsize=(6, 5))
    ax3 = sns.barplot(data=df, x='Risk_Label', y='P_human', hue='Risk_Label', errorbar=None, palette='Blues',
                      legend=False)
    plt.title('[H3] Tác động của Mức độ rủi ro đến Hành vi', fontweight='bold')
    plt.ylabel('Tỷ lệ chọn Con người (P_human)')
    plt.ylim(0, 1.2)
    for container in ax3.containers: ax3.bar_label(container, fmt='%.2f', padding=3)
    plt.tight_layout()
    plt.savefig('Chart_03_H3_Risk_Phuman.png', dpi=300)
    plt.close()

    # CHART 4: H4 (Tính chủ quan -> Hành vi)
    plt.figure(figsize=(6, 5))
    ax4 = sns.barplot(data=df, x='Subj_Label', y='P_human', hue='Subj_Label', errorbar=None, palette='Oranges',
                      legend=False)
    plt.title('[H4] Tác động của Lĩnh vực câu hỏi đến Hành vi', fontweight='bold')
    plt.ylabel('Tỷ lệ chọn Con người (P_human)')
    plt.ylim(0, 1.2)
    for container in ax4.containers: ax4.bar_label(container, fmt='%.2f', padding=3)
    plt.tight_layout()
    plt.savefig('Chart_04_H4_Subj_Phuman.png', dpi=300)
    plt.close()

    # CHART 5: H5 (Am hiểu AI điều tiết Rủi ro -> Hành vi)
    plt.figure(figsize=(7, 5))
    ax5 = sns.barplot(data=df, x='Risk_Label', y='P_human', hue='Mức độ Am hiểu AI (Lit)', errorbar=None,
                      palette='viridis')
    plt.title('[H5] Mức độ am hiểu AI điều tiết (Risk -> Hành vi)', fontweight='bold')
    plt.ylabel('Tỷ lệ chọn Con người (P_human)')
    plt.ylim(0, 1.2)
    for container in ax5.containers: ax5.bar_label(container, fmt='%.2f', padding=3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('Chart_05_H5_Moderation_Risk_Phuman.png', dpi=300)
    plt.close()

    # CHART 6: H6 (Am hiểu AI điều tiết Tính chủ quan -> Hành vi)
    plt.figure(figsize=(7, 5))
    ax6 = sns.barplot(data=df, x='Subj_Label', y='P_human', hue='Mức độ Am hiểu AI (Lit)', errorbar=None,
                      palette='magma')
    plt.title('[H6] Mức độ am hiểu AI điều tiết (Subj -> Hành vi)', fontweight='bold')
    plt.ylabel('Tỷ lệ chọn Con người (P_human)')
    plt.ylim(0, 1.2)
    for container in ax6.containers: ax6.bar_label(container, fmt='%.2f', padding=3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('Chart_06_H6_Moderation_Subj_Phuman.png', dpi=300)
    plt.close()

    # CHART 7: H7 (Niềm tin -> Hành vi)
    # Lưu ý: Nếu Analysis.py chạy bị lỗi cột Trust thì đổi 'Trust' -> 'Trust_Norm' nhé.
    plt.figure(figsize=(6, 5))
    sns.regplot(data=df, x='Trust', y='P_human',
                scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
    plt.title('[H7] Tác động của Niềm tin nền tảng đến Hành vi', fontweight='bold')
    plt.xlabel('Mức độ Tin cậy Con người')
    plt.ylabel('Tỷ lệ chọn Con người (P_human)')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.tight_layout()
    plt.savefig('Chart_07_H7_Trust_Phuman.png', dpi=300)
    plt.close()

    print("Đã xuất thành công 7 biểu đồ theo mô hình Tác động Trực tiếp & Điều tiết!")