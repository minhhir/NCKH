import pandas as pd
import numpy as np

final = pd.read_csv('final_data.csv')
synthetic = pd.read_csv('synthetic_data.csv')

# Mapping scenario to question names
scenario_names = {
    0: "Q1_Supplier",
    1: "Q2_Paint", 
    2: "Q3_Machine",
    3: "Q4_Flight",
    4: "Q5_Lunch",
    5: "Q6_Spa",
    6: "Q7_Server",
    7: "Q8_Post",
    8: "Q9_Bridge",
    9: "Q10_Stock",
    10: "Q11_Vaccine",
    11: "Q12_Tower",
    12: "Q13_News",
    13: "Q14_Embezzlement",
    14: "Q15_Airbag",
    15: "Q16_Hostage"
}

print("=" * 100)
print("SO SÁNH P_HUMAN (TỈ LỆ CHỌN CON NGƯỜI) CỦA MỖI CÂU HỎI")
print("=" * 100)

comparison = []
for scenario_id in range(16):
    q_name = scenario_names.get(scenario_id, f"Q{scenario_id+1}")
    
    final_p_human = final[final['Scenario_ID'] == scenario_id]['P_human'].mean()
    synthetic_p_human = synthetic[synthetic['Scenario_ID'] == scenario_id]['P_human'].mean()
    
    diff = abs(final_p_human - synthetic_p_human)
    
    comparison.append({
        'Scenario': scenario_id,
        'Question': q_name,
        'Final_P_human': final_p_human,
        'Synthetic_P_human': synthetic_p_human,
        'Difference': diff
    })
    
    print(f"\n{q_name} (Scenario {scenario_id}):")
    print(f"  Final:     {final_p_human:.4f} ({final_p_human*100:.1f}% chọn Con người)")
    print(f"  Synthetic: {synthetic_p_human:.4f} ({synthetic_p_human*100:.1f}% chọn Con người)")
    print(f"  Khác biệt: {diff:.4f} ({diff*100:.1f} điểm)")

print("\n" + "=" * 100)
print("THỐNG KÊ TỔNG HỢP")
print("=" * 100)

df_comparison = pd.DataFrame(comparison)
print(f"\nKhác biệt trung bình: {df_comparison['Difference'].mean():.4f}")
print(f"Khác biệt lớn nhất:  {df_comparison['Difference'].max():.4f} ({df_comparison.loc[df_comparison['Difference'].idxmax(), 'Question']})")
print(f"Khác biệt nhỏ nhất:  {df_comparison['Difference'].min():.4f} ({df_comparison.loc[df_comparison['Difference'].idxmin(), 'Question']})")

print("\nCác câu có khác biệt >0.1 (nguy hiểm):")
large_diff = df_comparison[df_comparison['Difference'] > 0.1].sort_values('Difference', ascending=False)
for idx, row in large_diff.iterrows():
    print(f"  {row['Question']}: {row['Difference']:.4f} (Final: {row['Final_P_human']:.1%} vs Synthetic: {row['Synthetic_P_human']:.1%})")

print("\nSorted by Difference:")
print(df_comparison.sort_values('Difference', ascending=False)[['Question', 'Final_P_human', 'Synthetic_P_human', 'Difference']].to_string(index=False))
