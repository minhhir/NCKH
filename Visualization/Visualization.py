import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def _get_output_name(base_name, file_prefix):
    return f"{file_prefix}_{base_name}" if file_prefix else base_name


def visualize_results(df, include_trust=True, file_prefix=""):

    if df is None or df.empty:
        print("Không có dữ liệu để vẽ biểu đồ.")
        return

    sns.set_theme(style="whitegrid", font_scale=1.1)

    df = df.copy()
    df['Risk_Label'] = df['Risk'].map({0.0: 'Rủi ro Thấp', 1.0: 'Rủi ro Cao'})
    df['Subj_Label'] = df['Subj'].map({0.0: 'Khách quan', 1.0: 'Chủ quan'})
    df['Info_Label'] = df['Info'].map({0.0: 'Tải thấp', 1.0: 'Tải cao'})

    # Vì Lit đã được đưa về thang [0, 1] (0, 0.25, 0.5, 0.75, 1)
    # Mức 4, 5 tương đương >= 0.75
    df['Lit_Group'] = df['Lit'].apply(lambda x: 'Am hiểu Cao' if x >= 0.75 else 'Am hiểu Thấp/TB')

    median_d = df['D_total'].median()
    df['Conflict_Level'] = df['D_total'].apply(lambda x: 'Mâu thuẫn Cao' if x > median_d else 'Mâu thuẫn Thấp')

    # CHART 1: H1
    plt.figure(figsize=(6, 6))
    ax1 = sns.barplot(data=df, x='Conflict_Level', y='P_human', hue='Conflict_Level', palette='pastel', legend=False, errorbar=None)
    plt.title('H1: Tác động của Cường độ Mâu thuẫn lên Hành vi', fontweight='bold')
    plt.ylabel('Xác suất chọn Con người (P_human)')
    plt.xlabel('Mức độ Mâu thuẫn (D_total)')
    plt.ylim(0, 1.1)
    for i in ax1.containers: ax1.bar_label(i, fmt='%.2f', padding=3)
    plt.tight_layout()
    plt.savefig(_get_output_name('Chart_01_H1_Conflict.png', file_prefix), dpi=300)
    plt.close()

    # CHART 2: H2
    plt.figure(figsize=(6, 6))
    ax2 = sns.barplot(data=df, x='Info_Label', y='P_human', hue='Info_Label', palette='viridis', legend=False, errorbar=None)
    plt.title('H2: Tác động của Tải lượng thông tin lên Hành vi', fontweight='bold')
    plt.ylabel('Xác suất chọn Con người (P_human)')
    plt.xlabel('Tải lượng thông tin')
    plt.ylim(0, 1.1)
    for i in ax2.containers: ax2.bar_label(i, fmt='%.2f', padding=3)
    plt.tight_layout()
    plt.savefig(_get_output_name('Chart_02_H2_Info_Behavior.png', file_prefix), dpi=300)
    plt.close()

    # CHART 3: H3
    plt.figure(figsize=(8, 6))
    ax3 = sns.barplot(data=df, x='Risk_Label', y='P_human', hue='Risk_Label', palette='Set2', legend=False, errorbar=None)
    plt.title('H3: Tác động của Mức độ rủi ro lên Hành vi', fontweight='bold')
    plt.ylabel('Xác suất chọn Con người (P_human)')
    plt.xlabel('Mức độ Rủi ro')
    plt.ylim(0, 1.1)
    plt.axhline(0.5, linestyle='--', color='gray', alpha=0.5)
    for i in ax3.containers: ax3.bar_label(i, fmt='%.2f', padding=3)
    plt.tight_layout()
    plt.savefig(_get_output_name('Chart_03_H3_Risk_Behavior.png', file_prefix), dpi=300)
    plt.close()

    # CHART 4: H4
    plt.figure(figsize=(6, 6))
    ax4 = sns.barplot(data=df, x='Subj_Label', y='P_human', hue='Subj_Label', palette='Set3', legend=False, errorbar=None)
    plt.title('H4: Tác động của Lĩnh vực câu hỏi lên Hành vi', fontweight='bold')
    plt.ylabel('Xác suất chọn Con người (P_human)')
    plt.xlabel('Lĩnh vực vấn đề')
    plt.ylim(0, 1.1)
    for i in ax4.containers: ax4.bar_label(i, fmt='%.2f', padding=3)
    plt.tight_layout()
    plt.savefig(_get_output_name('Chart_04_H4_Subj_Behavior.png', file_prefix), dpi=300)
    plt.close()

    # CHART 5: H5 (Risk x Lit)
    plt.figure(figsize=(8, 6))
    sns.pointplot(data=df, x='Risk_Label', y='P_human', hue='Lit_Group', dodge=True, palette='Purples', markers=['o', 's'], linestyles=['-', '--'])
    plt.title('H5: Am hiểu AI điều tiết Rủi ro lên Hành vi', fontweight='bold')
    plt.ylabel('Xác suất chọn Con người (P_human)')
    plt.xlabel('Mức độ Rủi ro')
    plt.ylim(0, 1.1)
    plt.axhline(0.5, linestyle=':', color='gray', alpha=0.7)
    plt.legend(title='Am hiểu AI', loc='upper left')
    plt.tight_layout()
    plt.savefig(_get_output_name('Chart_05_H5_Risk_Lit_Behavior.png', file_prefix), dpi=300)
    plt.close()

    # CHART 6: H6 (Subj x Lit)
    plt.figure(figsize=(8, 6))
    sns.pointplot(data=df, x='Subj_Label', y='P_human', hue='Lit_Group', dodge=True, palette='Oranges', markers=['^', 'v'])
    plt.title('H6: Am hiểu AI điều tiết Lĩnh vực câu hỏi lên Hành vi', fontweight='bold')
    plt.ylabel('Xác suất chọn Con người (P_human)')
    plt.xlabel('Lĩnh vực vấn đề')
    plt.ylim(0, 1.1)
    plt.legend(title='Am hiểu AI')
    plt.tight_layout()
    plt.savefig(_get_output_name('Chart_06_H6_Subj_Lit_Behavior.png', file_prefix), dpi=300)
    plt.close()

    if include_trust and 'Trust_Norm' in df.columns:
        # CHART 7: H7
        plt.figure(figsize=(8, 6))
        sns.regplot(data=df, x='Trust_Norm', y='P_human', logistic=True, scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
        plt.title('H7: Tác động của Mức độ tin cậy lên Hành vi', fontweight='bold')
        plt.ylabel('Xác suất chọn Con người (P_human)')
        plt.xlabel('Mức độ tin cậy (Trust_Norm)')
        plt.ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.savefig(_get_output_name('Chart_07_H7_Trust_Behavior.png', file_prefix), dpi=300)
        plt.close()

        print("-> Đã tạo và lưu thành công 7 biểu đồ (có Trust).")
    else:
        print("-> Đã tạo và lưu thành công 6 biểu đồ (không Trust).")