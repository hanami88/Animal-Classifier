# CLIP Pet Classifier — Setup Guide

Tài liệu hướng dẫn cài đặt và khởi chạy dự án tại local.

## Yêu cầu hệ thống
- **Python** >= 3.10
- **Node.js** >= 20.19
- **Docker Desktop** (chạy PostgreSQL)
- **macOS 14+ on Apple Silicon** (required for the pinned MPS build)

---

## 1. Khởi động Database

Khởi chạy PostgreSQL qua Docker:

```bash
docker compose up -d
```

---

## 2. Cài đặt & Chạy Backend (FastAPI)

```bash
cd backend

# Khởi tạo và kích hoạt virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows
# source venv/bin/activate    # macOS/Linux

# Cài đặt thư viện
pip install -r requirements.txt

# Cấu hình biến môi trường
cp .env.example .env
# Chú ý: Hãy mở file .env và điền thông tin tài khoản Cloudinary của bạn:
# CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET

# Chạy DB migrations và seed dữ liệu ban đầu
alembic upgrade head
cd .. && python scripts/seed_breeds.py

# Xuất / Huấn luyện trọng số các Model AI (Bắt buộc để chạy inference)
python scripts/export_weights.py
cd backend

# Khởi chạy server
uvicorn app.main:app --reload --port 8000
```

> **Lưu ý lần chạy đầu:** Khi suy luận lần đầu tiên, backend tự động tải trọng số mô hình tách nền `DeepLabV3-MobileNetV3-Large` (~42 MB) từ PyTorch Hub nên cần có kết nối mạng (các lần sau dùng cache local; nếu tải/chạy tách nền lỗi sẽ tự fallback sang ảnh thô).

### Apple Silicon / MPS

The backend automatically selects Apple MPS when it is available. You can set
`AI_DEVICE=mps` in `backend/.env` to require MPS, or `AI_DEVICE=cpu` to force a
CPU fallback while diagnosing a Metal issue.

Verify the environment before starting the API:

```bash
python3 -c "import torch; print(torch.backends.mps.is_built(), torch.backends.mps.is_available())"
```

Both values should be `True` when MPS is required. Start the backend on macOS
with the same application command after activating `backend/venv`:

```bash
source venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

---

## 3. Cài đặt & Chạy Frontend (React Vite)

Mở một terminal mới:

```bash
cd frontend

# Cài đặt dependencies
npm install

# Khởi chạy ứng dụng
npm run dev
```

---

## 4. Kiến trúc hệ thống & Luồng hoạt động

Dự án **CLIP Pet Classifier** được xây dựng theo kiến trúc Client-Server hiện đại, kết hợp Xử lý bất đồng bộ (Asynchronous Background Task) để tối ưu việc chạy các Model AI nặng.

### Cấu trúc Frontend (React + Vite)
- Cung cấp giao diện tương tác mượt mà: Kéo thả ảnh (Drag & Drop), chọn Model AI, và xem kết quả biểu đồ trực quan.
- Sử dụng cơ chế Polling (gọi API liên tục mỗi 1 giây) để lắng nghe trạng thái của quá trình phân loại AI đang chạy ẩn dưới nền Backend.
- Lưu và hiển thị lịch sử nhận diện (History Dashboard) của người dùng.

### Cấu trúc Backend (FastAPI + PostgreSQL + Cloudinary)
1. **Upload & Storage**: Khi nhận được ảnh từ Client, Backend lập tức upload ảnh đó lên hệ thống lưu trữ đám mây **Cloudinary**. Cloudinary trả về một đường link URL an toàn.
2. **Database Tracking**: Backend lưu URL của bức ảnh, thời gian, tên Model đã chọn và trạng thái khởi tạo (`PROCESSING`) vào CSDL **PostgreSQL**.
3. **Background Inference**: Để không chặn luồng chính (không bắt người dùng chờ HTTP Request lâu), FastAPI giao nhiệm vụ chạy Model AI cho một Background Task.
4. **Phản hồi**: Background Task tải ảnh từ Cloudinary URL vào bộ nhớ RAM (`bytes`), đưa vào Model AI đã chọn, đo lường thời gian (Processing Time) và cập nhật kết quả Top-K (5 giống gần nhất) vào Database với trạng thái `COMPLETED`. Frontend nhờ cơ chế Polling sẽ nhận được kết quả này và hiển thị cho người dùng.

---

## 5. Hệ thống Models AI (CLIP-based)

Dự án tích hợp 4 mô hình học máy dựa trên nền tảng **OpenAI CLIP**, được thiết kế để phân loại 37 giống Chó & Mèo (từ Oxford-IIIT Pet Dataset) với các phương pháp và mức độ phức tạp khác nhau:

- **1. CLIP Zero-Shot Baseline**: 
  - Mô hình gốc không qua huấn luyện (Training-Free).
  - So sánh trực tiếp ảnh tải lên với các câu prompt (mô tả bằng văn bản).
  - Tốc độ cực nhanh nhưng độ chính xác thấp nhất (87.76%).
  
- **2. Tip-Adapter-F (Cache Adapter)**: 
  - Sử dụng cơ chế Few-Shot Learning, tạo một bộ nhớ đệm (Cache Model) chứa đặc trưng của các ảnh đã học.
  - Kết hợp (blend) giữa trọng số Zero-shot và dự đoán từ thuật toán KNN (K-Nearest Neighbors).
  - Tăng độ chính xác lên 89.56%.
  
- **3. CLIP-Adapter (Feature Adapter)**: 
  - Đóng băng toàn bộ trọng số của CLIP, chỉ huấn luyện một mạng Neural tuyến tính nhỏ (Bottleneck MLP) bổ sung vào cuối bộ trích xuất đặc trưng (Feature Extractor).
  - Tốc độ huấn luyện nhanh, dung lượng siêu nhẹ gọn (chỉ vài chục KB) nhưng đạt độ chính xác ấn tượng (90.27%).
  
- **4. CoOp (Context Optimization - Prompt Tuning)**:
  - Thay vì tự viết prompt thủ công (vd: *"a photo of a..."*), mô hình này tiến hành huấn luyện các bộ *vector ngữ cảnh liên tục* (Learnable Context Vectors) để ghép vào văn bản đầu vào.
  - Đây là Model mạnh mẽ nhất trong dự án, xử lý tốt ngữ nghĩa và đạt độ chính xác cao nhất (91.61%).

*Lưu ý: Tất cả các Model trong dự án đều tích hợp cơ chế **TTA (Test-Time Augmentation)**: tự động cắt ảnh ban đầu thành 5 mảnh (4 góc + trung tâm) để tăng cường độ ổn định và chính xác khi dự đoán thực tế.*
