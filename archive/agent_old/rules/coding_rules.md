# Quy định lập trình (Coding Rules) - CLIP Animal Classifier

Tệp tài liệu này định nghĩa các quy tắc lập trình, quản lý tài nguyên và cấu trúc mã nguồn bắt buộc phải tuân thủ trong quá trình thực thi dự án.

---

## 1. Quản lý Thiết bị (Device Management)

Mã nguồn chạy huấn luyện và đánh giá cần tự động phát hiện thiết bị phần cứng để tối ưu hóa hiệu năng, đặc biệt là hỗ trợ Apple Silicon (MPS):

```python
import torch

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
```

> [!IMPORTANT]
> Tránh việc gán cứng thiết bị là `"cuda"` hoặc `"cpu"`. Luôn sử dụng hàm phát hiện tự động để mã nguồn tương thích cả trên macOS (MPS) và GPU Linux/Windows.

---

## 2. Quản lý Dữ liệu (Dataset Management)

* **Tập dữ liệu sử dụng**: Oxford-IIIT Pet dataset từ thư viện `torchvision.datasets`.
* **Đường dẫn lưu trữ**: Dữ liệu phải được tải xuống và lưu trữ thống nhất tại thư mục `./dataset/` ở gốc dự án để tránh việc tải lại nhiều lần.
* **Xử lý tiền xử lý (Preprocessing)**:
  * Sử dụng các transform tương ứng với mô hình CLIP. Đối với CLIP Hugging Face, sử dụng `CLIPProcessor`.
  * Đảm bảo tập support (few-shot) và tập query/test được tách biệt hoàn toàn (không rò rỉ dữ liệu).

---

## 3. Cấu trúc Module hóa Mã nguồn

Mã nguồn phải được tách biệt thành các file/module chức năng cụ thể:
1. `data_utils.py`: Chứa mã nguồn tải dữ liệu, tiền xử lý, stratified sampling để tạo Few-shot support set.
2. `models.py`: Định nghĩa/khởi tạo mô hình ResNet baseline, tải mô hình CLIP từ Hugging Face.
3. `adaptation.py`: Chứa thuật toán huấn luyện Linear Probe CLIP và xây dựng Cache cho Tip-Adapter.
4. `evaluate.py`: Thực hiện đo đạc các chỉ số: Acc per shot, Latency, Confidence Interval và vẽ Confusion Matrix.

---

## 4. Quản lý Log Thực nghiệm (Experiment Logging)

* Mọi thí nghiệm chạy xong cần log lại kết quả dưới dạng file JSON hoặc lưu vào tensorboard/mlflow (nếu có).
* Các kết quả đo lường được lưu trong thư mục `results/` với định dạng tên: `results_{method}_{shots}shot.json` chứa:
  * `accuracy`: Giá trị accuracy trung bình.
  * `confidence_interval`: `[lower_bound, upper_bound]`.
  * `latency`: Thống kê độ trễ (mean, p50, p95, p99).
  * `confusion_matrix`: Dữ liệu ma trận dưới dạng array.
