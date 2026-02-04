import pandas as pd
import statsmodels.formula.api as smf


def run_analysis(df):

    # Phase 2: Dự báo Hành vi (Xác định các Beta)
    model_ph = smf.ols("WOA ~ Trust + Conflict + Risk + Conflict:Risk", data=df).fit()
    betas = model_ph.params

    df['Calculated_WOA'] = model_ph.predict(df)
    # xuất tóm tắt mô hình hồi quy ra file txt


    with open('statistical_summary_phase2.txt', 'w', encoding='utf-8') as f:
        f.write("DU BAO HANH VI\n")
        f.write(model_ph.summary().as_text())

    df.to_csv('recalculated_research_data.csv', index=False)
    return df


if __name__ == "__main__":
    df = pd.read_csv('cleaned_research_data_final.csv')
    run_analysis(df)
