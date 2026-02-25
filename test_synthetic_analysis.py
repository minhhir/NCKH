import pandas as pd
from Analysis.Analysis import run_analysis

# Load synthetic data
df_synthetic = pd.read_csv('synthetic_data.csv')

print("=" * 80)
print("KIỂM ĐỊNH GIẢ THUYẾT TRÊN DỮ LIỆU SYNTHETIC (CÓ TRUST)")
print("=" * 80)

# Run with trust
run_analysis(
    df_synthetic,
    include_trust=True,
    output_file="Synthetic_Results_With_Trust.txt"
)

print("\n" + "=" * 80)
print("KIỂM ĐỊNH GIẢ THUYẾT TRÊN DỮ LIỆU SYNTHETIC (KHÔNG TRUST)")
print("=" * 80)

# Run without trust
run_analysis(
    df_synthetic,
    include_trust=False,
    output_file="Synthetic_Results_Without_Trust.txt"
)

print("\n✓ Kết quả được lưu vào:")
print("  - Synthetic_Results_With_Trust.txt")
print("  - Synthetic_Results_Without_Trust.txt")
