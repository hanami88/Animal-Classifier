# Pet Breed Classifier — Hệ thống nhận diện 37 giống chó và mèo

Website phân loại giống thú cưng (**37 giống chó và mèo** thuộc bộ dữ liệu Oxford-IIIT Pet: 12 giống mèo, 25 giống chó) sử dụng **5 mô hình mạng nơ-ron tích chập (CNN) tự huấn luyện**, với **Dual-Stream EfficientNet-B0** (hai luồng Thân + Mặt kết hợp cơ chế cổng Gated Fusion) là mô hình đề xuất chính và mặc định.

---

## 1. Bảng 5 mô hình & Độ chính xác công bố

Tất cả 5 mô hình được huấn luyện từ đầu (from scratch) trên tập mở rộng 9.301 ảnh và đánh giá trên 3.669 ảnh chuẩn `Oxford-IIIT Pet test` (với kích thước đầu vào `288 × 288`):

| Mô hình (`method`) | Nhóm kiến trúc | Số tham số | Top-1 (Single-view) | Top-1 (4-View TTA) | Ghi chú |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Dual-Stream EfficientNet-B0** (`dual-stream-efficientnet`) | Hai luồng (Thân + Mặt) | 5,042,081 | **85.72%** | **86.54%** | **Đề xuất chính / Mặc định** |
| **EfficientNet-B0** (`efficientnet-b0`) | Một luồng | 4,045,605 | 83.32% | 83.78% | Cân bằng tốt tốc độ & độ chính xác |
| **DenseNet-121** (`densenet121`) | Một luồng | 6,991,781 | 74.54% | 75.33% | Tái sử dụng đặc trưng đa tầng |
| **SEResNet-18** (`seresnet18`) | Một luồng | 11,282,533 | 70.62% | 71.55% | ResNet-18 kèm Squeeze-and-Excitation |
| **ResNet-18** (`resnet18`) | Một luồng | 11,195,493 | 67.84% | 68.47% | Mô hình cơ sở đối chiếu (Baseline) |

---

## 2. Kiến trúc hệ thống

Hệ thống vận hành theo mô hình **Client–Server bất đồng bộ**:

1. **Frontend (React + TypeScript + Vite)**:
   - Giao diện tải ảnh kéo-thả / chọn từ thư viện hoặc chụp ảnh trên điện thoại (`JPG`, `PNG`, `WebP`, tối đa `10 MB`).
   - Cho phép chọn trực tiếp 1 trong 5 mô hình hoặc chế độ **All Models** để chạy cả 5 mô hình song song trên cùng 1 ảnh và so sánh độ đồng thuận / bất đồng Top-1.
   - Công tắc **"Độ chính xác cao (TTA)"** mặc định tắt (bật 4-View TTA với trọng số `0.40 / 0.30 / 0.15 / 0.15`, hậu tố `+tta` được lưu ở `model_version`).
   - Trang **History** (lọc theo trạng thái & mô hình) và **Statistics** (biểu đồ hoạt động, tỷ lệ loài/giống, độ tin cậy, độ trễ).
2. **Backend (FastAPI + PyTorch + Torchvision + Alembic)**:
   - Kiểm tra định dạng, dung lượng (`<= 10 MB`), giải mã hợp lệ bằng Pillow và kích thước ảnh (`<= 8000 px`) trước khi lưu.
   - Đưa xử lý suy luận AI vào `BackgroundTasks`; ở chế độ **All Models**, bước tách nền bằng `DeepLabV3` chỉ chạy **1 lần duy nhất** rồi chia sẻ kết quả tiền xử lý cho cả 5 mô hình.
   - Nạp mô hình lười (lazy-load), khóa an toàn luồng và `load_state_dict(..., strict=True)`.
3. **PostgreSQL (Docker)**: Lưu 3 bảng `species`, `breeds` (37 giống) và `predictions` (trạng thái `UPLOADED` / `PROCESSING` / `COMPLETED` / `FAILED`, điểm tin cậy Top-5 và thời gian xử lý).
4. **Cloudinary**: Lưu trữ ảnh người dùng tải lên và trả URL an toàn để hiển thị lại ở trang Lịch sử / Thống kê.

---

## 3. Cấu trúc thư mục

```text
.
├── backend/
│   ├── alembic/                  # DB migrations (PostgreSQL)
│   ├── app/
│   │   ├── ai/                   # 5 kiến trúc CNN, tiền xử lý DeepLabV3 + TTA, Registry
│   │   ├── api/routes/           # REST API (/models, /predictions, /prediction-history, /statistics, /breeds)
│   │   ├── core/                 # Config (.env) và kết nối SQLAlchemy
│   │   ├── models/ & schemas/    # ORM models & Pydantic schemas
│   │   ├── repositories/         # Truy vấn PostgreSQL
│   │   ├── services/             # Chạy suy luận nền cho Single và Batch
│   │   └── storage/              # Upload ảnh lên Cloudinary
│   ├── weights/                  # Chứa 5 file *_best.pth và 5 file *_state_dict.pth (gitignored)
│   ├── .env.example              # Mẫu biến môi trường (không chứa khoá thật)
│   └── requirements.txt          # Thư viện runtime cho Web Backend
├── frontend/                     # Ứng dụng React + TypeScript + Vite
├── my_training/                  # Mã nguồn, 5 notebook và requirements huấn luyện gốc
├── scripts/
│   ├── extract_state_dicts.py    # Trích xuất *_state_dict.pth nhẹ từ *_best.pth
│   ├── seed_breeds.py            # Nạp danh sách 2 loài & 37 giống vào PostgreSQL
│   └── verify_new_models.py      # Kiểm thử 5 model trên tập test Oxford-IIIT Pet
├── docs/                         # Tài liệu phân tích (AUDIT.md) và quyết định kỹ thuật (DECISIONS.md)
├── docker-compose.yml            # Dịch vụ PostgreSQL 16
└── SETUP.md                      # Hướng dẫn cài đặt từng bước từ máy sạch
```

---

## 4. Hạn chế đã biết

- **Chênh lệch Trimap khi suy luận thực tế:** Lúc huấn luyện/đánh giá trên tập `Oxford-IIIT Pet`, tiền cảnh được cắt chính xác bằng nhãn `Trimap` thật và XML Bounding Box phần đầu. Khi người dùng tải ảnh ngoài đời thực (không có Trimap), hệ thống dùng mô hình `deeplabv3_mobilenet_v3_large` để tách nền tự động (`cat`/`dog`) và tô nền xám `(128, 128, 128)`, còn luồng đầu của `Dual-Stream` dùng vùng cắt quy tắc (`65%` phía trên, `10%–90%` chiều ngang).
  - Đo thực nghiệm nội bộ trên 300 ảnh `Oxford-IIIT Pet test` cho thấy tách nền tự động đạt **84.00% / 85.00%** (Single/TTA) với `Dual-Stream EfficientNet-B0` và **79.00% / 82.33%** với `EfficientNet-B0`, thấp hơn nhẹ so với dùng Trimap chuẩn trong notebook.
- **Ảnh ngoài phân phối Oxford:** Với ảnh thú cưng chụp quá xa, bị che khuất, nhiều con vật chồng lấn, hoặc khi DeepLab không phát hiện được `cat`/`dog` (hệ thống tự fallback về ảnh thô để tránh lỗi request), độ chính xác có thể giảm so với benchmark công bố.
- **Độ trễ khi bật TTA và lần chạy đầu:** Chế độ **4-View TTA** tăng thời gian xử lý gấp ~3–4 lần; ngoài ra request suy luận đầu tiên sau khi khởi động server sẽ chậm hơn do nạp lazy mô hình `DeepLabV3` và các file weights vào bộ nhớ.

---

Để cài đặt và chạy dự án trên máy local, xem hướng dẫn chi tiết tại [SETUP.md](SETUP.md).
