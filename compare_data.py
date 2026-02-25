import pandas as pd
import numpy as np
from scipy import stats

final = pd.read_csv('final_data.csv')
synthetic = pd.read_csv('synthetic_data.csv')

print("=" * 80)
print("PHÂN TÍCH D_TOTAL")
print("=" * 80)

print("\n[FINAL DATA] D_total:")
print(f"  Mean: {final['D_total'].mean():.6f}")
print(f"  Std:  {final['D_total'].std():.6f}")
print(f"  Min:  {final['D_total'].min():.6f}")
print(f"  Max:  {final['D_total'].max():.6f}")
print(f"  Unique values: {final['D_total'].nunique()}")
print(f"  Sample values: {sorted(final['D_total'].unique())[:20]}")

print("\n[SYNTHETIC DATA] D_total:")
print(f"  Mean: {synthetic['D_total'].mean():.6f}")
print(f"  Std:  {synthetic['D_total'].std():.6f}")
print(f"  Min:  {synthetic['D_total'].min():.6f}")
print(f"  Max:  {synthetic['D_total'].max():.6f}")
print(f"  Unique values: {synthetic['D_total'].nunique()}")
print(f"  Sample values: {sorted(synthetic['D_total'].unique())[:20]}")

print("\n" + "=" * 80)
print("PHÂN TÍCH P_HUMAN")
print("=" * 80)

print("\n[FINAL DATA] P_human:")
print(final['P_human'].value_counts().sort_index())
print(f"  P_human = 0: {(final['P_human'] == 0).sum()} ({(final['P_human'] == 0).mean()*100:.1f}%)")
print(f"  P_human = 1: {(final['P_human'] == 1).sum()} ({(final['P_human'] == 1).mean()*100:.1f}%)")

print("\n[SYNTHETIC DATA] P_human:")
print(synthetic['P_human'].value_counts().sort_index())
print(f"  P_human = 0: {(synthetic['P_human'] == 0).sum()} ({(synthetic['P_human'] == 0).mean()*100:.1f}%)")
print(f"  P_human = 1: {(synthetic['P_human'] == 1).sum()} ({(synthetic['P_human'] == 1).mean()*100:.1f}%)")

print("\n" + "=" * 80)
print("PHÂN BỐ P_HUMAN THEO SCENARIO")
print("=" * 80)

print("\n[FINAL] P_human by Scenario_ID:")
final_by_scenario = final.groupby(['Scenario_ID'])['P_human'].agg(['mean', 'std', 'count'])
print(final_by_scenario)

print("\n[SYNTHETIC] P_human by Scenario_ID:")
synthetic_by_scenario = synthetic.groupby(['Scenario_ID'])['P_human'].agg(['mean', 'std', 'count'])
print(synthetic_by_scenario)

print("\n" + "=" * 80)
print("KIỂM TRA SKEWNESS VÀ PHÂN BỐ")
print("=" * 80)

print(f"\n[FINAL] P_human skewness: {stats.skew(final['P_human']):.4f}")
print(f"[SYNTHETIC] P_human skewness: {stats.skew(synthetic['P_human']):.4f}")

print(f"\n[FINAL] P_human kurtosis: {stats.kurtosis(final['P_human']):.4f}")
print(f"[SYNTHETIC] P_human kurtosis: {stats.kurtosis(synthetic['P_human']):.4f}")
