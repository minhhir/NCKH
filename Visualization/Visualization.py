import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


def visualize_results(df):
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # =========================================================
    # CHART 0: BIỂU ĐỒ NHIỆT MA TRẬN TƯƠNG QUAN (HEATMAP)
    # Rất phổ biến trong các bài báo khoa học chuẩn mực
    # =========================================================
    var_names = ['Ctx', 'Risk', 'Subj', 'Info', 'AILit', 'Trust', 'DV']
    df_corr = df[var_names].dropna()
    corr = df_corr.corr()

    # Tạo mask để chỉ hiển thị tam giác dưới
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Tạo mảng text annotation (chỉ số + sao ý nghĩa)
    annot_labels = np.empty_like(corr, dtype=object)
    for i in range(len(var_names)):
        for j in range(len(var_names)):
            if i > j:
                val = corr.iloc[i, j]
                _, pval = pearsonr(df_corr[var_names[i]], df_corr[var_names[j]])
                stars = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
                annot_labels[i, j] = f"{val:.2f}{stars}"
            else:
                annot_labels[i, j] = ""

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, mask=mask, annot=annot_labels, fmt='', cmap='coolwarm',
                vmax=1, vmin=-1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .7})
    plt.title('Ma trận tương quan (Correlation Heatmap)', fontweight='bold')
    plt.tight_layout()
    plt.savefig('Chart_00_Correlation_Heatmap.png', dpi=300)
    plt.close()

    # =========================================================
    # CÁC BIỂU ĐỒ CÒN LẠI (GIỮ NGUYÊN)
    # =========================================================
    df['Risk_Label'] = df['Risk'].map({0.0: 'Rủi ro Thấp', 1.0: 'Rủi ro Cao'})
    df['Subj_Label'] = df['Subj'].map({0.0: 'Khách quan', 1.0: 'Chủ quan'})
    df['Info_Label'] = df['Info'].map({0.0: 'Tải thấp', 1.0: 'Tải cao'})
    df['Mức độ Am hiểu AI (AILit)'] = df['AILit'].apply(lambda x: 'Am hiểu Cao' if x >= 0.75 else 'Am hiểu Thấp/TB')

    median_d = df['Ctx'].median()
    df['Conflict_Level'] = df['Ctx'].apply(lambda x: 'Mâu thuẫn Cao' if x > median_d else 'Mâu thuẫn Thấp')

    # CHART 1: H1 (Bối cảnh Ctx -> DV)
    plt.figure(figsize=(6, 5))
    ax1 = sns.barplot(data=df, x='Conflict_Level', y='DV', hue='Conflict_Level', errorbar=None, palette='pastel',
                      legend=False)
    plt.title('[H1] Tác động của Bối cảnh đến Hành vi', fontweight='bold')
    plt.ylabel('Hành vi chấp nhận lời khuyên (DV)')
    plt.ylim(0, 1.0)
    for container in ax1.containers: ax1.bar_label(container, fmt='%.2f', padding=3)
    plt.tight_layout()
    plt.savefig('Chart_01_H1_Ctx_DV.png', dpi=300)
    plt.close()

    # CHART 2: H2 (Tải lượng Info -> DV)
    plt.figure(figsize=(6, 5))
    ax2 = sns.barplot(data=df, x='Info_Label', y='DV', hue='Info_Label', errorbar=None, palette='Set2', legend=False)
    plt.title('[H2] Tác động của Tải lượng thông tin đến Hành vi', fontweight='bold')
    plt.ylabel('Hành vi chấp nhận lời khuyên (DV)')
    plt.ylim(0, 1.0)
    for container in ax2.containers: ax2.bar_label(container, fmt='%.2f', padding=3)
    plt.tight_layout()
    plt.savefig('Chart_02_H2_Info_DV.png', dpi=300)
    plt.close()

    # CHART 3: H3 (Rủi ro Risk -> DV)
    plt.figure(figsize=(6, 5))
    ax3 = sns.barplot(data=df, x='Risk_Label', y='DV', hue='Risk_Label', errorbar=None, palette='Blues', legend=False)
    plt.title('[H3] Tác động của Mức độ rủi ro đến Hành vi', fontweight='bold')
    plt.ylabel('Hành vi chấp nhận lời khuyên (DV)')
    plt.ylim(0, 1.0)
    for container in ax3.containers: ax3.bar_label(container, fmt='%.2f', padding=3)
    plt.tight_layout()
    plt.savefig('Chart_03_H3_Risk_DV.png', dpi=300)
    plt.close()

    # CHART 4: H4 (Tính chủ quan Subj -> DV)
    plt.figure(figsize=(6, 5))
    ax4 = sns.barplot(data=df, x='Subj_Label', y='DV', hue='Subj_Label', errorbar=None, palette='Oranges', legend=False)
    plt.title('[H4] Tác động của Lĩnh vực câu hỏi đến Hành vi', fontweight='bold')
    plt.ylabel('Hành vi chấp nhận lời khuyên (DV)')
    plt.ylim(0, 1.0)
    for container in ax4.containers: ax4.bar_label(container, fmt='%.2f', padding=3)
    plt.tight_layout()
    plt.savefig('Chart_04_H4_Subj_DV.png', dpi=300)
    plt.close()

    # CHART 5: H5 (AILit điều tiết Risk -> DV)
    plt.figure(figsize=(7, 5))
    ax5 = sns.barplot(data=df, x='Risk_Label', y='DV', hue='Mức độ Am hiểu AI (AILit)', errorbar=None,
                      palette='viridis')
    plt.title('[H5] Mức độ am hiểu AI điều tiết (Risk -> Hành vi)', fontweight='bold')
    plt.ylabel('Hành vi chấp nhận lời khuyên (DV)')
    plt.ylim(0, 1.0)
    for container in ax5.containers: ax5.bar_label(container, fmt='%.2f', padding=3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('Chart_05_H5_Moderation_Risk_DV.png', dpi=300)
    plt.close()

    # CHART 6: H6 (AILit điều tiết Subj -> DV)
    plt.figure(figsize=(7, 5))
    ax6 = sns.barplot(data=df, x='Subj_Label', y='DV', hue='Mức độ Am hiểu AI (AILit)', errorbar=None, palette='magma')
    plt.title('[H6] Mức độ am hiểu AI điều tiết (Subj -> Hành vi)', fontweight='bold')
    plt.ylabel('Hành vi chấp nhận lời khuyên (DV)')
    plt.ylim(0, 1.0)
    for container in ax6.containers: ax6.bar_label(container, fmt='%.2f', padding=3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('Chart_06_H6_Moderation_Subj_DV.png', dpi=300)
    plt.close()

    # CHART 7: H7 (Niềm tin Trust -> DV)
    plt.figure(figsize=(6, 5))
    sns.regplot(data=df, x='Trust', y='DV', scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
    plt.title('[H7] Tác động của Niềm tin nền tảng đến Hành vi', fontweight='bold')
    plt.xlabel('Mức độ Tin cậy Con người')
    plt.ylabel('Hành vi chấp nhận lời khuyên (DV)')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.tight_layout()
    plt.savefig('Chart_07_H7_Trust_DV.png', dpi=300)
    plt.close()

    # CHART 8: INTERACTION PLOT (Risk x AILit)
    plt.figure(figsize=(7, 5))
    ax8 = sns.pointplot(data=df, x='Risk_Label', y='DV', hue='Mức độ Am hiểu AI (AILit)',
                        markers=['o', 's'], linestyles=['-', '--'], palette='viridis', errorbar=None)
    plt.title('[H5] Biểu đồ Tương tác (Interaction Plot): Risk x AILit', fontweight='bold')
    plt.xlabel('Mức độ Rủi ro (Risk)')
    plt.ylabel('Hành vi chấp nhận lời khuyên (DV)')
    plt.ylim(0, 1.0)
    plt.legend(title='Mức độ Am hiểu AI', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('Chart_08_Interaction_Risk_AILit.png', dpi=300)
    plt.close()

    print("Đã vẽ biểu đồ nhiệt và xuất thành công 9 biểu đồ!")