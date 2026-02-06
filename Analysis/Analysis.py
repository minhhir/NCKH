import statsmodels.formula.api as smf


def run_analysis(df):
    #  Phân tích Niềm tin
    #Trust = α0 + α1(Risk) + α2(Subj) + α3(Risk x Subj) + ϵ1
    model_trust = smf.ols("Trust_Base ~ Risk + Subj + Risk:Subj + Literacy", data=df).fit()

    #Phân tích Hành vi
    # Logit(PHuman) = β0 + β1*(Risk) + β2*(Subj) + β3*(Literacy) + β4*(Literacy*Risk) + β5*(Subj*Risk)
    formula_woa = "WOA ~ Risk + Subj + Literacy + Literacy:Risk + Subj:Risk"
    model_woa = smf.logit(formula_woa, data=df).fit(method='bfgs', maxiter=200, disp=0)

    with open('statistical_summary_final.txt', 'w', encoding='utf-8') as f:
        f.write("[MODEL 1] NIỀM TIN (H4)\n" + model_trust.summary().as_text() + "\n\n")
        f.write("[MODEL 2] HÀNH VI WOA (H1, H3, H5)\n" + model_woa.summary().as_text())
    return df, model_woa