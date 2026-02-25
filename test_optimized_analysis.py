import pandas as pd
from Analysis.Analysis import run_analysis

# Load optimized synthetic data
df_optimized = pd.read_csv('synthetic_data_hypothesis_optimized.csv')

print("=" * 80)
print("KIỂM ĐỊNH GIẢ THUYẾT - SYNTHETIC OPTIMIZED (CÓ TRUST)")
print("=" * 80)

# Run with trust
run_analysis(
    df_optimized,
    include_trust=True,
    output_file="Synthetic_Results_Optimized_With_Trust.txt"
)

print("\n" + "=" * 80)
print("KIỂM ĐỊNH GIẢ THUYẾT - SYNTHETIC OPTIMIZED (KHÔNG TRUST)") 
print("=" * 80)

# Run without trust
run_analysis(
    df_optimized,
    include_trust=False,
    output_file="Synthetic_Results_Optimized_Without_Trust.txt"
)
