import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def visualize_results(df):

    # Cấu hình giao diện chuẩn học thuật
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # Chuẩn bị dữ liệu vẽ biểu đồ
    df['Risk_Label'] = df['Risk'].map({0.0: 'Rủi ro Thấp', 1.0: 'Rủi ro Cao'})
    df['Subj_Label'] = df['Subj'].map({0.0: 'Khách quan', 1.0: 'Chủ quan'})
    df['Info_Label'] = df['Info'].map({0.0: 'Tải thấp', 1.0: 'Tải cao'})

    # Chia nhóm Literacy (Am hiểu AI) theo thang 1-5
    # Mức 4, 5: Am hiểu Cao | Mức 1, 2, 3: Am hiểu Thấp/Trung bình
    df['Lit_Group'] = df['Lit'].apply(lambda x: 'Am hiểu Cao (Mức 4-5)' if x >= 4 else 'Am hiểu Thấp/TB (Mức 1-3)')

    # Chia nhóm D_total (Cường độ Mâu thuẫn) qua trung vị
    median_d = df['D_total'].median()
    df['Conflict_Level'] = df['D_total'].apply(lambda x: 'Mâu thuẫn Cao' if x > median_d else 'Mâu thuẫn Thấp')

    # ==========================================
    # CHART 1: H1 (Mâu thuẫn -> Hành vi)
    plt.figure(figsize=(6, 6))
    ax1 = sns.barplot(data=df, x='Conflict_Level', y='P_human', hue='Conflict_Level', palette='pastel', legend=False, errorbar=None)
    plt.title('H1: Tác động của Cường độ Mâu thuẫn lên Hành vi', fontweight='bold')
    plt.ylabel('Xác suất chọn Con người (P_human)')
    plt.xlabel('Mức độ Mâu thuẫn (D_total)')
    plt.ylim(0, 1.1)
    for i in ax1.containers: ax1.bar_label(i, fmt='%.2f', padding=3)
    plt.tight_layout()
    plt.savefig('Chart_01_H1_Conflict.png', dpi=300)
    plt.close()

    # ==========================================
    # CHART 2: H2 (Niềm tin -> Hành vi)
    plt.figure(figsize=(8, 6))
    sns.regplot(data=df, x='Trust_Norm', y='P_human', logistic=True, scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
    plt.title('H2: Tác động của Niềm tin lên Hành vi lựa chọn', fontweight='bold')
    plt.ylabel('Xác suất chọn Con người (P_human)')
    plt.xlabel('Niềm tin vào AI (Trust_Norm)')
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig('Chart_02_H2_Trust_Behavior.png', dpi=300)
    plt.close()

    # ==========================================
    # CHART 3: H3 (Rủi ro x Chủ quan -> Hành vi)
    plt.figure(figsize=(8, 6))
    ax3 = sns.barplot(data=df, x='Risk_Label', y='P_human', hue='Subj_Label', palette='Set2', errorbar=None)
    plt.title('H3: Rủi ro khuếch đại tính Chủ quan (Risk x Subj)', fontweight='bold')
    plt.ylabel('Xác suất chọn Con người (P_human)')
    plt.xlabel('Mức độ Rủi ro')
    plt.ylim(0, 1.1)
    plt.axhline(0.5, linestyle='--', color='gray', alpha=0.5)
    for i in ax3.containers: ax3.bar_label(i, fmt='%.2f', padding=3)
    plt.legend(title='Lĩnh vực vấn đề', loc='upper right')
    plt.tight_layout()
    plt.savefig('Chart_03_H3_Risk_Subj_Phuman.png', dpi=300)
    plt.close()

    # ==========================================
    # CHART 4: H4 (Chủ quan -> Niềm tin)
    plt.figure(figsize=(6, 6))
    ax4 = sns.barplot(data=df, x='Subj_Label', y='Trust_Norm', hue='Subj_Label', palette='Set3', legend=False, errorbar=None)
    plt.title('H4 (a,b): Tính chủ quan tác động lên Niềm tin', fontweight='bold')
    plt.ylabel('Mức độ Niềm tin vào AI (Trust)')
    plt.xlabel('Lĩnh vực vấn đề')
    plt.ylim(0, 1.1)
    for i in ax4.containers: ax4.bar_label(i, fmt='%.2f', padding=3)
    plt.tight_layout()
    plt.savefig('Chart_04_H4_Subj_Trust.png', dpi=300)
    plt.close()

    # ==========================================
    # CHART 5A: H5a (Rủi ro x Am hiểu AI -> Niềm tin)
    plt.figure(figsize=(8, 6))
    sns.pointplot(data=df, x='Risk_Label', y='Trust_Norm', hue='Lit_Group', dodge=True, palette='Blues', markers=['o', 's'])
    plt.title('H5a: Am hiểu AI điều tiết Rủi ro lên Niềm tin', fontweight='bold')
    plt.ylabel('Mức độ Niềm tin vào AI (Trust)')
    plt.xlabel('Mức độ Rủi ro')
    plt.ylim(0, 1.05)
    plt.legend(title='Am hiểu AI (Lit)', loc='lower right')
    plt.tight_layout()
    plt.savefig('Chart_05a_H5a_Risk_Lit_Trust.png', dpi=300)
    plt.close()

    # ==========================================
    # CHART 5B: H5b (Rủi ro x Am hiểu AI -> Hành vi)
    plt.figure(figsize=(8, 6))
    sns.pointplot(data=df, x='Risk_Label', y='P_human', hue='Lit_Group', dodge=True, palette='Purples', markers=['o', 's'], linestyles=['-', '--'])
    plt.title('H5b: Am hiểu AI điều tiết Rủi ro lên Hành vi', fontweight='bold')
    plt.ylabel('Xác suất chọn Con người (P_human)')
    plt.xlabel('Mức độ Rủi ro')
    plt.ylim(0, 1.1)
    plt.axhline(0.5, linestyle=':', color='gray', alpha=0.7)
    plt.legend(title='Am hiểu AI (Lit)', loc='upper left')
    plt.tight_layout()
    plt.savefig('Chart_05b_H5b_Risk_Lit_Phuman.png', dpi=300)
    plt.close()

    # ==========================================
    # CHART 6: H6 (Tải thông tin -> Hành vi)
    plt.figure(figsize=(6, 6))
    ax6 = sns.barplot(data=df, x='Info_Label', y='P_human', hue='Info_Label', palette='viridis', legend=False, errorbar=None)
    plt.title('H6: Tác động của Tải lượng thông tin (Info Load)', fontweight='bold')
    plt.ylabel('Xác suất chọn Con người (P_human)')
    plt.xlabel('Tải lượng thông tin')
    plt.ylim(0, 1.1)
    for i in ax6.containers: ax6.bar_label(i, fmt='%.2f', padding=3)
    plt.tight_layout()
    plt.savefig('Chart_06_H6_InfoLoad.png', dpi=300)
    plt.close()

    # ==========================================
    # CHART 7: H7 (Chủ quan x Am hiểu AI -> Niềm tin)
    plt.figure(figsize=(8, 6))
    sns.pointplot(data=df, x='Subj_Label', y='Trust_Norm', hue='Lit_Group', dodge=True, palette='Oranges', markers=['^', 'v'])
    plt.title('H7: Am hiểu AI điều tiết Tính chủ quan lên Niềm tin', fontweight='bold')
    plt.ylabel('Mức độ Niềm tin vào AI (Trust)')
    plt.xlabel('Lĩnh vực vấn đề')
    plt.ylim(0, 1.05)
    plt.legend(title='Am hiểu AI (Lit)')
    plt.tight_layout()
    plt.savefig('Chart_07_H7_Subj_Lit_Trust.png', dpi=300)
    plt.close()

    # ==========================================
    # CHART 8: H8 (Rủi ro x Tải thông tin -> Hành vi)
    plt.figure(figsize=(8, 6))
    ax8 = sns.barplot(data=df, x='Risk_Label', y='P_human', hue='Info_Label', palette='magma', errorbar=None)
    plt.title('H8: Tải thông tin suy yếu Rủi ro (Risk x Info)', fontweight='bold')
    plt.ylabel('Xác suất chọn Con người (P_human)')
    plt.xlabel('Mức độ Rủi ro')
    plt.ylim(0, 1.1)
    for i in ax8.containers: ax8.bar_label(i, fmt='%.2f', padding=3)
    plt.legend(title='Tải lượng thông tin', loc='upper right')
    plt.tight_layout()
    plt.savefig('Chart_08_H8_Risk_Info_Phuman.png', dpi=300)
    plt.close()

    # ==========================================
    # CHART 9: H9 (Trung gian: Tác động tổng của Chủ quan lên Hành vi)
    plt.figure(figsize=(6, 6))
    ax9 = sns.barplot(data=df, x='Subj_Label', y='P_human', hue='Subj_Label', palette='muted', legend=False, errorbar=None)
    plt.title('H9 (Total Effect): Tính chủ quan tác động Hành vi', fontweight='bold')
    plt.ylabel('Xác suất chọn Con người (P_human)')
    plt.xlabel('Lĩnh vực vấn đề')
    plt.ylim(0, 1.1)
    for i in ax9.containers: ax9.bar_label(i, fmt='%.2f', padding=3)
    plt.tight_layout()
    plt.savefig('Chart_09_H9_Total_Effect.png', dpi=300)
    plt.close()

    # ==========================================
    # CHART 10: H10 (Rủi ro x Chủ quan -> Niềm tin)
    plt.figure(figsize=(8, 6))
    ax10 = sns.barplot(data=df, x='Risk_Label', y='Trust_Norm', hue='Subj_Label', palette='coolwarm', errorbar=None)
    plt.title('H10: Rủi ro điều tiết Tính chủ quan lên Niềm tin', fontweight='bold')
    plt.ylabel('Mức độ Niềm tin vào AI (Trust)')
    plt.xlabel('Mức độ Rủi ro')
    plt.ylim(0, 1.1)
    for i in ax10.containers: ax10.bar_label(i, fmt='%.2f', padding=3)
    plt.legend(title='Lĩnh vực vấn đề', loc='upper right')
    plt.tight_layout()
    plt.savefig('Chart_10_H10_Risk_Subj_Trust.png', dpi=300)
    plt.close()

    print("-> Đã tạo và lưu thành công 11 Biểu đồ KHÔNG CẢNH BÁO, CHỈN CHU, SẮC NÉT!")