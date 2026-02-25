import pandas as pd
import numpy as np

SYNTHETIC_FILE = "synthetic_data.csv"
FINAL_DATA_FILE = "final_data.csv"
OUTPUT_FILE = "synthetic_responses.csv"

# Age groups and genders from Form nghiên cứu
AGE_GROUPS = [
    "Dưới 18",
    "Từ 18 đến 24",
    "Từ 25 đến 34",
    "Từ 35 đến 50",
    "Trên 50"
]

GENDERS = ["Nam", "Nữ"]

# Distribution: Gen Z (18-24) focused, then <18, minimal for others
AGE_WEIGHTS = [0.12, 0.80, 0.03, 0.03, 0.02]  # <18, 18-24, 25-34, 35-50, 50+
GENDER_WEIGHTS = [0.5, 0.5]  # Equal male/female


def create_synthetic_responses():
    """Convert synthetic_data.csv (long format) to response format (wide format) similar to Form nghiên cứu."""
    
    # Load synthetic data
    synthetic_df = pd.read_csv(SYNTHETIC_FILE)
    final_df = pd.read_csv(FINAL_DATA_FILE)
    
    # Get user info (demographics) from final_data for original users only
    user_info = final_df[["User_ID", "Lit", "Trust_Norm"]].drop_duplicates(subset=["User_ID"])
    
    # Pivot synthetic data to wide format: rows = users, cols = scenarios
    pivot_df = synthetic_df.pivot_table(
        index="User_ID",
        columns="Scenario_ID",
        values="P_human",
        aggfunc="first"
    )
    
    # Rename scenario columns to descriptive names
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
        15: "Q16_Hostage",
    }
    pivot_df = pivot_df.rename(columns=scenario_names)
    
    # Convert P_human (0/1) to response text
    for col in scenario_names.values():
        pivot_df[col] = pivot_df[col].apply(
            lambda x: "Lời khuyên AI" if x == 0.0 else ("Lời khuyên Con người" if x == 1.0 else None)
        )
    
    # Reset index to make User_ID a column
    pivot_df = pivot_df.reset_index()
    
    # Merge with user info
    result_df = pivot_df.merge(user_info, on="User_ID", how="left")
    
    # Add age group and gender
    np.random.seed(42)
    result_df["Age_Group"] = np.random.choice(AGE_GROUPS, size=len(result_df), p=AGE_WEIGHTS)
    result_df["Gender"] = np.random.choice(GENDERS, size=len(result_df), p=GENDER_WEIGHTS)
    
    # Reorder columns: User_ID, demographics, then scenarios
    demo_cols = ["User_ID", "Age_Group", "Gender", "Lit", "Trust_Norm"]
    scenario_cols = list(scenario_names.values())
    result_df = result_df[demo_cols + scenario_cols]
    
    # Save to CSV
    result_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"✅ Created: {OUTPUT_FILE}")
    print(f"   - Users: {len(result_df)}")
    print(f"   - Scenarios: {len(scenario_cols)}")
    print(f"\nAge Group Distribution:")
    print(result_df["Age_Group"].value_counts())
    print(f"\nGender Distribution:")
    print(result_df["Gender"].value_counts())
    print(f"\nFirst 5 rows:")
    print(result_df.head())


if __name__ == "__main__":
    create_synthetic_responses()
