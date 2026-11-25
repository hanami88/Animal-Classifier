---
name: model_adaptation
description: Hướng dẫn kỹ thuật triển khai các thuật toán Zero-shot, Linear Probe, và Tip-Adapter bằng CLIP.
---

# Kỹ năng thích ứng mô hình (Model Adaptation Skills)

Tài liệu này hướng dẫn chi tiết cách triển khai ba phương pháp phân loại giống vật nuôi bằng mô hình CLIP: Zero-shot CLIP, Linear Probe CLIP, và Tip-Adapter.

---

## 1. Zero-shot CLIP (Baseline & Prompt Engineering)

### Nguyên lý
So sánh trực tiếp cosine similarity giữa vector đặc trưng ảnh (image embedding) và vector đặc trưng văn bản (text embedding) được sinh ra từ các prompt mô tả nhãn lớp.

### Các bước triển khai
1. Tải mô hình CLIP từ Hugging Face:
   ```python
   from transformers import CLIPModel, CLIPProcessor
   model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
   processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
   ```
2. Trích xuất Image features và Text features (đã chuẩn hóa L2):
   ```python
   # Normalize features
   image_features = image_features / image_features.norm(dim=-1, keepdim=True)
   text_features = text_features / text_features.norm(dim=-1, keepdim=True)
   ```
3. Tính logits và dự đoán:
   ```python
   # Similarity matrix
   logits = image_features @ text_features.t() * model.logit_scale.exp()
   predictions = logits.argmax(dim=-1)
   ```

---

## 2. Linear Probe CLIP

### Nguyên lý
Đóng băng toàn bộ các trọng số của CLIP (gồm cả Image Encoder và Text Encoder). Chỉ sử dụng Image Encoder làm bộ trích xuất đặc trưng (feature extractor) cố định. Sau đó, huấn luyện một bộ phân loại tuyến tính đơn giản (ví dụ: một lớp `nn.Linear` hoặc Logistic Regression) trên tập dữ liệu support (Few-shot).

### Các bước triển khai
1. Trích xuất đặc trưng và lưu trữ (để tăng tốc độ huấn luyện):
   ```python
   with torch.no_grad():
       support_features = clip_model.get_image_features(support_images)
       support_features = support_features / support_features.norm(dim=-1, keepdim=True)
   ```
2. Định nghĩa classifier tuyến tính:
   ```python
   import torch.nn as nn
   
   classifier = nn.Linear(in_features=support_features.shape[1], out_features=num_classes)
   optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
   criterion = nn.CrossEntropyLoss()
   ```
3. Huấn luyện trên tập support qua nhiều epoch để tối ưu trọng số lớp tuyến tính.

---

## 3. Tip-Adapter (Training-Free Few-Shot Adaption)

### Nguyên lý
Tip-Adapter xây dựng một bộ nhớ cache dạng Key-Value từ các ảnh support. 
* **Key**: Image embedding của các ảnh trong tập support.
* **Value**: Nhãn one-hot tương ứng của các ảnh support.
Khi suy diễn trên ảnh test, Tip-Adapter kết hợp kết quả Zero-shot logits truyền thống với kết quả thu hồi (retrieval logits) từ bộ nhớ cache. Phương pháp này hoàn toàn không cần huấn luyện lại mô hình (training-free) nhưng mang lại hiệu năng vượt trội.

### Công thức tính toán
Gọi $F_{test}$ là embedding của ảnh test, $F_{support}$ là embedding của tập support (kích thước $M \times D$), $W_{text}$ là embedding của văn bản prompt (kích thước $C \times D$), $L_{support}$ là ma trận nhãn one-hot của tập support (kích thước $M \times C$).

1. **Zero-shot logits**: 
   $$f_{zs} = F_{test} W_{text}^T$$
2. **Cache similarity**: 
   $$A = \exp(-\beta (1 - F_{test} F_{support}^T))$$
   Với $\beta$ là hệ số điều chỉnh tỷ lệ (sharpness).
3. **Final logits**:
   $$f_{final} = f_{zs} + \alpha A L_{support}$$
   Với $\alpha$ là trọng số tích hợp giữa phần Zero-shot và phần Cache.

### Các bước triển khai
```python
# F_test: (N_test, D), F_support: (M, D), W_text: (C, D), L_support: (M, C)
# Tắt autograd vì đây là phương pháp training-free

# Step 1: Zero-shot logits
logits_zs = F_test @ W_text.t() # (N_test, C)

# Step 2: Tính similarity với cache
# Cosine similarity matrix giữa test và support features
sim_matrix = F_test @ F_support.t() # (N_test, M)
A = torch.exp(-beta * (1.0 - sim_matrix)) # (N_test, M)

# Step 3: Kết hợp dự đoán
logits_final = logits_zs + alpha * (A @ L_support) # (N_test, C)
```
