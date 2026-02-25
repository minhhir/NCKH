import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load final data
df = pd.read_csv('final_data.csv')

# Fit model on final data to get true coefficients
formula = "P_human ~ D_total + Info + Risk + Subj + Lit + Risk:Lit + Subj:Lit + Trust_Norm"
model = smf.gee(
    formula=formula,
    groups="User_ID",
    data=df.dropna(subset=['P_human', 'D_total', 'Info', 'Risk', 'Subj', 'Lit', 'Trust_Norm']),
    family=sm.families.Binomial()
).fit()

print("=" * 80)
print("COEFFICIENTS FROM FINAL DATA (True Effect Sizes)")
print("=" * 80)
print(model.summary())

print("\n" + "=" * 80)
print("PYTHON CODE FOR SYNTHETIC_SCENARIO.PY:")
print("=" * 80)

print(f"""
    b_d_total    = {model.params['D_total']:.6f}    # H1: {model.bse['D_total']:.4f} (p={model.pvalues['D_total']:.4f})
    b_info       = {model.params['Info']:.6f}       # H2
    b_risk       = {model.params['Risk']:.6f}       # H3: {model.bse['Risk']:.4f} (p={model.pvalues['Risk']:.4f})
    b_subj       = {model.params['Subj']:.6f}       # H4: {model.bse['Subj']:.4f} (p={model.pvalues['Subj']:.4f})
    b_lit        = {model.params['Lit']:.6f}
    b_risk_lit   = {model.params['Risk:Lit']:.6f}   # H5
    b_subj_lit   = {model.params['Subj:Lit']:.6f}   # H6
    b_trust      = {model.params['Trust_Norm']:.6f}  # H7
""")

print("\n--- SUMMARY ---")
for col in ['D_total', 'Info', 'Risk', 'Subj', 'Lit', 'Risk:Lit', 'Subj:Lit', 'Trust_Norm']:
    print(f"{col:12} coef={model.params[col]:8.4f}  p={model.pvalues[col]:.4f}  {'✓' if model.pvalues[col] < 0.05 else '✗'}")
