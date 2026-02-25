from Data_Processing.Data_Processing import preprocess_data
from Analysis.Analysis import run_analysis
from Visualization.Visualization import visualize_results


def main():
    # Bước 1: Tiền xử lý
    df_clean =preprocess_data('Form nghiên cứu.csv')

    if df_clean is None or df_clean.empty:
        print("Không có dữ liệu hợp lệ để phân tích.")
        return

    # Bước 2A: Chạy mô hình có biến Trust
    df_with_trust = run_analysis(
        df_clean,
        include_trust=True,
        output_file="Statistical_Results_With_Trust.txt"
    )

    # Bước 2B: Chạy mô hình không có biến Trust
    df_without_trust = run_analysis(
        df_clean,
        include_trust=False,
        output_file="Statistical_Results_Without_Trust.txt"
    )

    # Bước 3: Xuất biểu đồ theo từng trường hợp
    visualize_results(df_with_trust, include_trust=True, file_prefix="WithTrust")
    visualize_results(df_without_trust, include_trust=False, file_prefix="WithoutTrust")

if __name__ == "__main__":
    main()
