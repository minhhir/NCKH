# 📊 Dự án Nghiên cứu: Mô hình Lựa chọn Tư vấn AI hoặc Con người theo Bối cảnh

## 📖 1. Tổng quan Lý thuyết & Mô hình Nghiên cứu
Dự án này là mã nguồn phục vụ cho nghiên cứu khoa học hành vi: **"Tác động của mức độ rủi ro và mâu thuẫn lời khuyên lên niềm tin và hành vi sử dụng tư vấn của người ra quyết định."**

Mục tiêu của nghiên cứu là tìm hiểu xem con người sẽ đặt "niềm tin trách nhiệm" vào Trí tuệ Nhân tạo (AI) hay Chuyên gia Con người khi đối mặt với các bối cảnh khác nhau (rủi ro cao/thấp, có/không có mâu thuẫn quan điểm, lĩnh vực câu hỏi, v.v.). 

### Sơ đồ Biến số (Variables)
* **Biến phụ thuộc (DV):** Hành vi chấp nhận lời khuyên (Giá trị: `0` = Chọn AI, `1` = Chọn Con người).
* **Biến độc lập (IVs):**
    * `Ctx` (Bối cảnh): `0` = Đồng thuận (Consensus), `1` = Mâu thuẫn (Conflict).
    * `Risk` (Mức độ rủi ro): `0` = Thấp, `1` = Cao.
    * `Subj` (Lĩnh vực): `0` = Khách quan (Toán/Logic), `1` = Chủ quan (Xã hội/Tình huống).
    * `Info` (Tải lượng thông tin): `0` = Thấp, `1` = Cao.
    * `Trust` (Niềm tin nền tảng): Giá trị `[0, 1]`.
* **Biến điều tiết (Moderator):** * `AILit` (Mức độ am hiểu AI): Giá trị `[0, 1]`. Đóng vai trò điều tiết tác động của `Risk` và `Subj` lên hành vi lựa chọn (`DV`).

## ⚙️ 2. Công nghệ & Phương pháp Thống kê

### Phương pháp tiếp cận
Do thiết kế nghiên cứu yêu cầu mỗi người tham gia phải trả lời nhiều kịch bản khác nhau (16 scenarios/người), các quan sát trên cùng một cá nhân là **không độc lập**. Do đó, dự án sử dụng:
* **Generalized Estimating Equations (GEE):** Mô hình phương trình ước lượng tổng quát với phân phối **Binomial** (do DV là biến nhị phân) và cấu trúc hiệp phương sai **Exchangeable** để kiểm soát phương sai nội nhóm (cluster effect) của từng người dùng.

### Tech Stack
* **Ngôn ngữ:** Python 3.x
* **Thư viện xử lý dữ liệu:** `pandas`, `numpy`
* **Thư viện thống kê & Machine Learning:** `statsmodels` (Chạy mô hình GEE, tính VIF), `scipy` (Tính p-value cho tương quan Pearson).
* **Thư viện Trực quan hóa:** `matplotlib`, `seaborn` (Vẽ biểu đồ nhiệt Heatmap, Barplot, Interaction Plot).

## 📂 3. Cấu trúc Thư mục

Dự án bao gồm các script Python và các file dữ liệu theo chuẩn pipeline khoa học dữ liệu:

* **File Code (.py):**
    * `AC_label.py`: Script tiền xử lý dữ liệu thô ban đầu để tính toán các nhãn bối cảnh (Ctx/D_total).
    * `Data_Processing.py`: Đọc dữ liệu thô, map các biến số từ 16 kịch bản, làm sạch text và chuyển đổi thành dạng Long-Format (`final_data.csv`).
    * `Analysis.py`: Chứa logic cốt lõi. Tính toán đa cộng tuyến (Ma trận tương quan, VIF) và chạy mô hình hồi quy GEE.
    * `Visualization.py`: Chịu trách nhiệm render hệ thống biểu đồ chuẩn học thuật.
    * `Main.py`: Script điều phối, tự động chạy chuỗi: Tiền xử lý -> Phân tích -> Trực quan hóa.
    * `toExcel.py`: Tiện ích hỗ trợ xuất dữ liệu ra file Excel phục vụ báo cáo.
* **File Dữ liệu Đầu vào (Inputs):**
    * `Form nghiên cứu.csv`: Dữ liệu khảo sát thô tải về từ hệ thống.
    * `QuestionForm_cleantext.xlsx - Sheet1.csv`: Dữ liệu đã được làm sạch text ban đầu.
    * `Ac_Results_Final.xlsx` (hoặc bản `.csv`): File Metadata chứa trọng số/nhãn của các kịch bản.
* **File Dữ liệu Đầu ra (Outputs):**
    * `final_data.csv`: Dữ liệu Long-Format đã làm sạch, sẵn sàng để đưa vào mô hình.
    * `GEE_Results.txt`: Kết quả thống kê chi tiết của mô hình.
    * `Correlation_VIF_Academic.csv`: Bảng ma trận tương quan và VIF định dạng chuẩn để copy vào Word.

## 🚀 4. Hướng dẫn Thực hành (Step-by-Step)

### Bước 1: Cài đặt Môi trường
Mở Terminal/Command Prompt và cài đặt các thư viện phụ thuộc:
```pip install pandas numpy statsmodels scipy seaborn matplotlib openpyxl```
Bước 2: Khởi tạo Metadata (Nếu cần)
Chạy script để xử lý nhãn bối cảnh từ file câu hỏi:

Bash
python AC_label.py
Bước 3: Chạy Pipeline Phân tích Chính
Chỉ cần chạy file Main.py, hệ thống sẽ tự động quét dữ liệu, làm sạch, chạy GEE và xuất ảnh:

Bash
python Main.py
Bước 4: Trích xuất Báo cáo
Sau khi chạy xong, hãy kiểm tra thư mục gốc:

Mở file Correlation_VIF_Academic.csv bằng Excel, copy bảng lưới và dán thẳng vào luận văn/báo cáo.

Mở GEE_Results.txt để lấy chỉ số Hệ số Beta (β) và P-value cho việc biện luận 7 giả thuyết (H1 đến H7).

Chèn các biểu đồ .png (Chart_00 đến Chart_08) vào phần Phụ lục hoặc Kết quả nghiên cứu.

📈 5. Diễn giải Kết quả (Interpreting the Output)
VIF (Variance Inflation Factor): Nếu các chỉ số trong cột VIF đều < 5 (hoặc < 10), dữ liệu không bị hiện tượng đa cộng tuyến nghiêm trọng.

P-value (P>|z|): Trong file GEE_Results.txt, giả thuyết được "Ủng hộ" (Supported) nếu p-value < 0.05 và dấu của hệ số Beta (Coef.) khớp với kỳ vọng ban đầu.

Interaction Plot (Chart_08): Biểu đồ đường chéo thể hiện sự tương tác. Nếu hai đường có độ dốc khác nhau rõ rệt hoặc cắt nhau, biến điều tiết (AILit) thực sự có tác động đến mối quan hệ giữa Rủi ro (Risk) và Hành vi (DV).
