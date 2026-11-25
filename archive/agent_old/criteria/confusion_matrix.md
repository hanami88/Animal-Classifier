# Tiêu chí: Ma trận nhầm lẫn (Confusion Matrix)

Tài liệu này đặc tả quy trình tạo lập, biểu diễn trực quan và phân tích lỗi phân loại bằng Ma trận nhầm lẫn cho 37 giống vật nuôi của tập dữ liệu Oxford-IIIT Pet.

---

## 1. Định nghĩa Confusion Matrix

Ma trận nhầm lẫn $C$ có kích thước $37 \times 37$. Trong đó:
* Phần tử $C_{i, j}$ thể hiện số lượng mẫu thực tế thuộc giống $i$ nhưng mô hình dự đoán nhầm sang giống $j$.
* **Chuẩn hóa (Normalization)**: Bắt buộc phải chuẩn hóa ma trận theo hàng (chia cho tổng số mẫu thực tế của từng lớp) để thể hiện tỉ lệ phần trăm ($0\% \to 100\%$). Điều này giúp loại bỏ sự chênh lệch về số lượng mẫu kiểm thử giữa các lớp.

---

## 2. Quy tắc Trực quan hóa (Visualization Rules)

Do số lượng lớp lớn (37 giống), việc vẽ biểu đồ heatmap cần tuân thủ:
1. **Kích thước ảnh lớn**: Đảm bảo `figsize` tối thiểu là `(16, 16)` inches để nhãn tên giống không bị chồng chéo lên nhau.
2. **Sắp xếp lớp theo loài (Species Grouping)**:
   * Sắp xếp danh sách nhãn sao cho 12 giống mèo đứng cạnh nhau và 25 giống chó đứng cạnh nhau.
   * Cách sắp xếp này giúp tạo ra hai vùng hình vuông lớn dọc theo đường chéo chính: vùng nhầm lẫn nội bộ giống mèo (Intra-cat errors) và vùng nhầm lẫn nội bộ giống chó (Intra-dog errors). Bất kỳ điểm sáng nào nằm ngoài hai vùng này sẽ lập tức chỉ ra lỗi nhầm lẫn chéo loài (Cross-species errors).
3. **Màu sắc (Colormap)**: Sử dụng bảng màu dễ quan sát (ví dụ `"Blues"`, `"viridis"`, hoặc `"rocket"`).

---

## 3. Phân tích lỗi (Error Analysis Workflow)

Khi chạy đánh giá, Agent cần trích xuất và phân tích:
* **Top 5 cặp giống vật nuôi bị nhầm lẫn nhiều nhất**: Ví dụ, giống mèo Bengal dễ bị nhầm với mèo Egyptian Mau do có họa tiết đốm tương tự; các giống chó Bulldog (French Bulldog, English Bulldog) dễ bị nhầm với nhau.
* **Tỉ lệ nhầm lẫn chéo loài (Cat vs. Dog)**: CLIP có nhầm lẫn chó thành mèo không? (Nhờ prompt engineering `"a type of [cat/dog]"`, tỷ lệ này thường phải giảm về mức cực thấp < 1%).

---

## 4. Minh họa code vẽ Confusion Matrix bằng Python

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, class_names, output_path="results/confusion_matrix.png"):
    """
    Tính toán, chuẩn hóa và vẽ confusion matrix.
    y_true, y_pred: list/array các nhãn
    class_names: list 37 tên lớp đã sắp xếp theo nhóm loài
    """
    # Tính confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    # Chuẩn hóa theo hàng
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(18, 16))
    sns.heatmap(
        cm_normalized, 
        annot=False, # Tắt annot số vì 37x37 quá chật chội
        fmt=".2f", 
        cmap="Blues", 
        xticklabels=class_names, 
        yticklabels=class_names
    )
    plt.ylabel('Thực tế (Ground Truth)')
    plt.xlabel('Dự đoán (Predicted)')
    plt.title('Normalized Confusion Matrix (Grouped by Species)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
```
