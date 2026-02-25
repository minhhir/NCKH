import pandas as pd
import os

CSV_FILES = [
    "final_data.csv",
    "synthetic_data.csv",
    "comparison_by_scenario.csv",
    "comparison_changed_rows.csv",
]


def convert_csv_to_xlsx():
    for csv_file in CSV_FILES:
        if not os.path.exists(csv_file):
            print(f"⚠️  {csv_file} not found, skipping...")
            continue

        try:
            df = pd.read_csv(csv_file)
            xlsx_file = csv_file.replace(".csv", ".xlsx")
            df.to_excel(xlsx_file, index=False, sheet_name="Data")
            print(f"✅ Converted: {csv_file} -> {xlsx_file}")
        except Exception as e:
            print(f"❌ Error converting {csv_file}: {e}")


if __name__ == "__main__":
    convert_csv_to_xlsx()
