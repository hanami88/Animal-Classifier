# Tiêu chí: Độ chính xác theo số mẫu hỗ trợ (Accuracy per Shot)

Tài liệu này đặc tả quy trình và cách thức tính toán chỉ số Accuracy per Shot nhằm so sánh khả năng học ít mẫu (Few-shot learning) của các mô hình.

---

## 1. Định nghĩa và Công thức

* **Số shot ($N_{shot}$)**: Số lượng ảnh hỗ trợ (support images) được sử dụng cho mỗi lớp giống vật nuôi trong quá trình thích ứng. Các mốc đánh giá bắt buộc: $N_{shot} \in \{1, 2, 4, 8, 16\}$.
* **Công thức Accuracy (độ chính xác)**:
  $$Accuracy = \frac{N_{correct}}{N_{total}} \times 100\%$$
  Trong đó:
  * $N_{correct}$: Số lượng ảnh trong tập test dự đoán đúng nhãn giống loài.
  * $N_{total}$: Tổng số lượng ảnh trong tập test (~3,700 ảnh).

---

## 2. Quy tắc Lấy mẫu (Sampling Guidelines)

Để đảm bảo kết quả đánh giá là khách quan và không bị thiên lệch (biased) bởi cách chọn mẫu:

1. **Stratified Sampling**: Khi lấy mẫu $N$ ảnh cho tập support, bắt buộc phải lấy đúng $N$ ảnh cho từng lớp trong số 37 lớp giống vật nuôi (tổng số ảnh support là $37 \times N$).
2. **Nhiều lượt chạy độc lập (Multiple runs)**: 
   * Không được chỉ chạy đánh giá 1 lần duy nhất trên 1 tập support ngẫu nhiên.
   * Quy định bắt buộc: Chạy thí nghiệm **10 lần độc lập** với 10 random seeds khác nhau để sinh ra 10 tập support khác nhau.
   * Kết quả báo cáo cuối cùng phải là **giá trị trung bình (mean)** của 10 lần chạy này kèm theo **độ lệch chuẩn (standard deviation)**.

---

## 3. Quy trình Đánh giá của Agent

Khi thực hiện nhiệm vụ đánh giá Accuracy, Agent cần:
* Thiết lập seed cố định cho từng lượt chạy (ví dụ từ seed 1 đến 10) để đảm bảo tính tái lập (reproducibility).
* Đánh giá và ghi nhận kết quả chi tiết cho từng phương pháp (ResNet, Zero-shot, LP-CLIP, Tip-Adapter) ở tất cả các mốc shot.
* Vẽ biểu đồ đường (line plot) thể hiện sự thay đổi của Accuracy theo số shot (trục hoành là số shot, trục tung là accuracy) để so sánh trực quan tốc độ cải thiện của từng phương pháp.
