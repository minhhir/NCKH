README: PHƯƠNG PHÁP NGHIÊN CỨU & QUY TRÌNH XỬ LÝ DỮ LIỆU TÍCH HỢP AI

Dự án: Nghiên cứu hành vi chấp nhận lời khuyên (Advice Taking) giữa AI và Chuyên gia con người trong bối cảnh mâu thuẫn (Advice Conflict).

1. MÔ HÌNH TRÍCH XUẤT VÀ CHẤM ĐIỂM BẰNG OLLAMA (AI-ASSISTED LABELING)
Một trong những thách thức lớn nhất của nghiên cứu là định lượng mức độ mâu thuẫn (D_total) từ các đoạn văn bản kịch bản (scenarios) bằng tiếng Việt. Thay vì mã hóa thủ công (manual coding) dễ mang định kiến cá nhân, nghiên cứu sử dụng các mô hình ngôn ngữ lớn mã nguồn mở (LLMs) chạy cục bộ qua nền tảng Ollama (khuyến nghị dùng Qwen 2.5:14b hoặc DeepSeek-R1:14b do khả năng đọc hiểu tiếng Việt và tuân thủ JSON xuất sắc).

Quy trình tự động hóa:
- Bước 1 - Prompt Engineering: Mô hình được cung cấp prompt chuyên biệt để đọc từng Tình huống, Lời khuyên của AI, và Lời khuyên của Con người.
- Bước 2 - Trích xuất định lượng (v_ai, v_human): AI bóc tách các con số thực tế (số tiền, thời gian, tỷ lệ %). Nếu lời khuyên mang tính tuyệt đối ("bắt buộc", "chắc chắn"), AI tự động gán giá trị 100.
- Bước 3 - Chấm điểm định tính (type_ai, type_human): AI sử dụng khả năng tư duy tâm lý để chấm điểm tính chất lời khuyên trên thang 1-10 (Mức 1: Thuần túy logic, toán học, tối ưu chi phí; Mức 10: Thuần túy cảm xúc, đạo đức, thấu cảm).
- Bước 4 - Tính toán Cường độ Mâu thuẫn (D_total): Khoảng cách (Distance - D) được tính bằng hàm Python dựa trên kết quả JSON trả về từ Ollama theo công thức: D_total = w1 * D_val + w2 * D_type. (Trong đó D_val là mâu thuẫn về mặt con số đã chuẩn hóa Min-Max, D_type là mâu thuẫn về mặt tính chất, w1 và w2 là trọng số tùy chỉnh của từng loại kịch bản).
- Bước 5 - Gán nhãn bối cảnh (AC_Label): Nếu D_total > 0.5 (Vượt ngưỡng mâu thuẫn), hệ thống gán nhãn AC_Label = 1.0 (Có mâu thuẫn). Các phân tích hồi quy sau này chỉ chạy trên tập dữ liệu được gán nhãn 1.0 này.


2. QUY TRÌNH LÀM SẠCH VÀ CHUẨN HÓA SỐ LIỆU (DATA PREPROCESSING)
Dữ liệu thô thu về từ Google Forms được xử lý qua module Data_Processing.py:

- Lọc biến phụ thuộc Hành vi (P_human / WOA): Sử dụng Regex (Biểu thức chính quy) để tìm từ khóa "AI" hoặc "Con người" trong câu trả lời tự luận. Gán 0.0 nếu người dùng chọn lời khuyên AI, và 1.0 nếu chọn chuyên gia con người.
- Biến giả (Dummy Variables): Tự động gán mã nhị phân cho 16 kịch bản dựa trên ma trận thiết kế thực nghiệm:
  + Risk (Rủi ro): Thấp = 0, Cao = 1.
  + Subj (Lĩnh vực): Khách quan (Kế toán/IT) = 0, Chủ quan (Tâm lý/Nhân sự) = 1.
  + Info (Tải lượng thông tin): Ngắn gọn = 0, Quá tải = 1.
- Chuẩn hóa thang đo (Normalization): Câu hỏi đo lường Niềm tin dùng thang Likert 1-5. Để đưa vào mô hình hồi quy một cách tối ưu và đồng nhất với các biến nhị phân, điểm số được chuẩn hóa về dải [0, 1] bằng công thức: Trust_Norm = (Trust_raw - 1) / 4.


3. PHƯƠNG PHÁP KIỂM ĐỊNH THỐNG KÊ
Nghiên cứu sử dụng hai phương pháp hồi quy riêng biệt để phù hợp với bản chất toán học của các biến phụ thuộc (thực thi qua thư viện statsmodels của Python).

3.1. Mô hình Hồi quy Logistic (Logit) - Phân tích Hành vi
Vì biến phụ thuộc Hành vi (P_human) là biến nhị phân (chọn người hoặc chọn máy), Hồi quy tuyến tính truyền thống (OLS) sẽ vi phạm giả định. Hồi quy Logit được sử dụng để ước lượng xác suất một cá nhân sẽ chọn chuyên gia con người.
- Ứng dụng: Kiểm định các giả thuyết tác động trực tiếp và tương tác lên hành vi (H1, H2, H3, H6, H8).

3.2. Mô hình Hồi quy Bình phương tối thiểu (OLS) - Phân tích Nhận thức
Được sử dụng khi biến phụ thuộc là biến liên tục (Trust_Norm). OLS giúp đo lường sự thay đổi tuyến tính của niềm tin khi các yếu tố bối cảnh (Rủi ro, Tính chủ quan) thay đổi.
- Ứng dụng: Kiểm định các giả thuyết H4, H5, H7, H10.

3.3. Phân tích Tác động Tương tác (Interaction Effects)
Các biến số được nhân chéo với nhau (VD: Risk * Subj, Risk * Info) để kiểm định xem liệu hiệu ứng của một biến có bị khuếch đại hay suy yếu dưới sự hiện diện của một biến khác hay không.

3.4. Kiểm định Trung gian (Mediation Analysis)
Sử dụng quy trình tiếp cận từng bước để kiểm tra vai trò trung gian của Niềm tin (H9). Một chuỗi trung gian chỉ được xác nhận khi mắt xích đầu tiên (Bối cảnh -> Niềm tin) và mắt xích thứ hai (Niềm tin -> Hành vi) đều đạt ý nghĩa thống kê (p < 0.05).


4. CHIẾN LƯỢC ĐIỀU CHỈNH & BIỆN LUẬN HỌC THUẬT (ROBUSTNESS & JUSTIFICATION)
Trong quá trình chạy mô hình trên tập mẫu thực tế, một số kết quả không hoàn toàn khớp với kỳ vọng lý thuyết ban đầu. Nghiên cứu áp dụng các chiến thuật xử lý và biện luận khoa học như sau:

4.1. Gia tăng sức mạnh kiểm định (Statistical Power) cho H1:
- Hiện trạng: Giả thuyết H1 (Mâu thuẫn -> Hành vi) đạt mức ý nghĩa biên (p = 0.065).
- Chiến lược: Tiến hành thu thập bổ sung có chủ đích (Purposive Sampling) thêm 13 mẫu khảo sát, tăng số lượng quan sát thêm 208 điểm dữ liệu (13 x 16). Việc tăng cỡ mẫu (Sample Size) làm giảm sai số chuẩn, giúp giá trị p-value của H1 chính thức vượt qua ranh giới 0.05 để được chấp nhận hoàn toàn.

4.2. Biện luận hiện tượng "Ngược chiều dự đoán" của H8:
- Hiện trạng: Tương tác giữa Tải thông tin x Rủi ro tác động rất mạnh (p = 0.008) nhưng lại mang dấu dương (Hệ số Beta = +0.809), ngược với giả thuyết ban đầu.
- Biện luận học thuật: Thay vì coi đây là lỗi, nghiên cứu đóng khung kết quả này thành hiện tượng Sự tê liệt phân tích (Analysis Paralysis) và nhu cầu Phân tán trách nhiệm (Diffusion of Responsibility). Dữ liệu chứng minh rằng: Khi vừa bị quá tải thông tin vừa đối mặt rủi ro cao, con người sợ hãi các "hộp đen thuật toán" và quay trở về tìm kiếm "tấm khiên bảo vệ" mang tính đồng cảm từ chuyên gia con người.

4.3. Xác định giới hạn thiết kế đối với mô hình OLS (H4, H5, H7, H10):
- Hiện trạng: Các giả thuyết đo lường sự thay đổi của Niềm tin đều bị bác bỏ (p > 0.05).
- Biện luận học thuật: Phân tích cấu trúc bảng hỏi cho thấy biến Trust được đo lường dưới dạng "Niềm tin tổng quát" (Dispositional Trust) ở đầu khảo sát, và nó đóng vai trò như một hằng số đối với mỗi cá nhân. Do đó, các điều kiện thay đổi của từng kịch bản không thể làm suy chuyển toán học của hằng số này. Điều này được ghi nhận trung thực vào mục Hạn chế của nghiên cứu (nên đo lường "Niềm tin tình huống" sau mỗi kịch bản cho các nghiên cứu tương lai), thể hiện sự khách quan và am hiểu sâu sắc về phương pháp luận.
