# Tiêu chí: Khoảng tin cậy của Accuracy (Confidence Interval)

Tài liệu này hướng dẫn cách tính toán Khoảng tin cậy 95% (95% Confidence Interval - CI) cho độ chính xác phân loại của mô hình, giúp kiểm chứng tính thuyết phục về mặt thống kê của các cải tiến.

---

## 1. Tại sao cần tính Khoảng tin cậy (CI)?

Khi so sánh hai phương pháp (ví dụ: CLIP Zero-shot đạt 80% và CLIP + Prompt Engineering đạt 87%):
* Khoảng tin cậy giúp xác định xem sự chênh lệch này có ý nghĩa thống kê (statistically significant) hay chỉ là do may rủi hoặc nhiễu trong dữ liệu kiểm thử.
* Nếu hai khoảng tin cậy 95% của hai mô hình không chồng chéo lên nhau (non-overlapping), ta có thể tự tin khẳng định mô hình có accuracy cao hơn thực sự tốt hơn.

---

## 2. Phương pháp thực hiện: Bootstrapping

Phương pháp phổ biến và mạnh mẽ nhất để ước lượng CI cho độ chính xác phân loại trên một tập test cố định là **Bootstrapping**:

### Thuật toán Bootstrapping
1. Lấy tập kết quả dự đoán đúng/sai trên Test Set của mô hình (kích thước $N_{test} \approx 3,700$ phần tử dạng nhị phân: 1 nếu đúng, 0 nếu sai).
2. Lặp lại $B = 1000$ lần (số lượt bootstrap):
   * Lấy mẫu ngẫu nhiên có hoàn lại (sampling with replacement) $N_{test}$ phần tử từ tập kết quả trên.
   * Tính toán accuracy trên mẫu bootstrap này.
3. Thu thập danh sách $B$ giá trị accuracy và sắp xếp theo thứ tự tăng dần.
4. Xác định khoảng tin cậy 95%:
   * **Giới hạn dưới (Lower Bound)**: Phân vị thứ 2.5 ($B \times 0.025 = 25$).
   * **Giới hạn trên (Upper Bound)**: Phân vị thứ 97.5 ($B \times 0.975 = 975$).

---

## 3. Minh họa code tính toán

Dưới đây là hàm Python chuẩn để Agent áp dụng tính toán CI:

```python
import numpy as np

def compute_confidence_interval(y_true, y_pred, num_bootstraps=1000, confidence_level=0.95):
    """
    Tính khoảng tin cậy bằng phương pháp Bootstrapping.
    y_true: nhãn thực tế
    y_pred: nhãn dự đoán của mô hình
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    is_correct = (y_true == y_pred).astype(float)
    
    bootstrapped_accuracies = []
    rng = np.random.default_rng(seed=42) # Đảm bảo tính nhất quán
    
    for _ in range(num_bootstraps):
        # Lấy mẫu ngẫu nhiên có hoàn lại
        bootstrap_indices = rng.choice(len(is_correct), size=len(is_correct), replace=True)
        acc = is_correct[bootstrap_indices].mean()
        bootstrapped_accuracies.append(acc)
        
    bootstrapped_accuracies = np.sort(bootstrapped_accuracies)
    
    # Tính percentile tương ứng với confidence level
    alpha = (1.0 - confidence_level) / 2.0
    lower_idx = int(num_bootstraps * alpha)
    upper_idx = int(num_bootstraps * (1.0 - alpha))
    
    lower_bound = bootstrapped_accuracies[lower_idx] * 100
    upper_bound = bootstrapped_accuracies[upper_idx] * 100
    
    return lower_bound, upper_bound
```
