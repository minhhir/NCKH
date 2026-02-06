import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def visualize_data(df):
    sns.set_theme(style="whitegrid")
    order = ['Thấp', 'Vừa', 'Cao']

    #  WOA theo Ngữ cảnh
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Risk_Label', y='WOA', hue='Subj_Label', order=order, palette='coolwarm')
    plt.title('Tỷ lệ chọn Con người theo Ngữ cảnh', fontweight='bold')
    plt.savefig('Chart_1_WOA_Summary.png')


    #  Literacy General
    plt.figure(figsize=(8, 5)); sns.regplot(data=df, x='Literacy', y=(1-df['WOA']), x_estimator=np.mean, logistic=True)
    plt.title(' Am hiểu AI và Xu hướng chọn AI', fontweight='bold');
    plt.savefig('Chart_2_Literacy_General.png')

    #Heatmap
    pivot = df.pivot_table(index='Subj_Label', columns='Risk_Label', values='WOA', aggfunc='mean').reindex(columns=order)
    plt.figure(figsize=(8, 5)); sns.heatmap(pivot, annot=True, cmap='YlGnBu');
    plt.title(' Bản đồ nhiệt ưu tiên Con người', fontweight='bold');
    plt.savefig('Chart_3_Heatmap.png')

    #  Groups Proportion
    df['Group'] = df['Risk_Label'] + " - " + df['Subj_Label']
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Group', y='WOA', hue='Group', palette='viridis', legend=False)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.title(' Tỷ lệ chọn theo 6 nhóm kết hợp', fontweight='bold');
    plt.savefig('Chart_4_Groups_Proportion.png')

    # Nghịch lý Rủi ro (H1 & H5)
    plt.figure(figsize=(10, 6))
    sns.pointplot(data=df, x='Risk_Label', y='WOA', hue='Subj_Label', order=order, palette='dark',linestyles=["-", "--"])
    plt.title('Nghịch lý Rủi ro và Vùng an toàn', fontweight='bold')
    plt.savefig('Chart_5_Risk_Paradox.png')

    # Trust Mechanism (H2)
    plt.figure(figsize=(8, 6)); sns.regplot(data=df, x='Trust_Base', y='WOA', logistic=True, line_kws={'color':'red'})
    plt.title(' Niềm tin dẫn dắt hành vi lựa chọn (H2)', fontweight='bold');
    plt.savefig('Chart_6_Trust_Mechanism.png')

    # Literacy Moderation (H5 Interaction)
    plt.figure(figsize=(10, 6)); df['Lit_Group'] = pd.cut(df['Literacy'], bins=[0, 3, 5], labels=['Thấp', 'Cao'])
    sns.pointplot(data=df, x='Risk_Label', y='WOA', hue='Lit_Group', order=order)
    plt.title(' Am hiểu AI điều tiết Rủi ro (H5)', fontweight='bold');
    plt.savefig('Chart_7_H5_Interaction.png')

    # Conflict Severity (D_Total)
    plt.figure(figsize=(10, 6)); sns.regplot(data=df, x='D_Total', y='WOA', logistic=True, color='purple')
    plt.title(' Mức độ mâu thuẫn (D_Total) và Lựa chọn', fontweight='bold');
    plt.savefig('Chart_8_Conflict_Severity.png')

    # Demographics
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(data=df, x='Gender', y='WOA', hue='Gender', ax=axes[0], palette='Pastel1', legend=False)
    sns.lineplot(data=df, x='Age', y='WOA', ax=axes[1], marker='o')
    plt.savefig('Chart_9_Demographics.png')