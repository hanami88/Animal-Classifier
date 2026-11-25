# Tiêu chí: Độ trễ suy diễn (Inference Latency)

Tài liệu này đặc tả quy trình kỹ thuật để đo lường chính xác tốc độ suy diễn (độ trễ) của các phương pháp phân loại vật nuôi trên cả CPU và thiết bị tăng tốc phần cứng (GPU/MPS).

---

## 1. Các quy tắc đo lường bắt buộc

Để phép đo thời gian suy diễn chính xác và phản ánh đúng thực tế hiệu năng hệ thống:

1. **Batch Size = 1**:
   * Phép đo latency suy diễn trên từng ảnh phải được thực hiện với batch size là 1, mô phỏng trực tiếp kịch bản thực tế khi hệ thống phân loại ảnh người dùng gửi lên theo thời gian thực (realtime).
2. **Warm-up phase (Giai đoạn khởi động)**:
   * Do PyTorch và GPU/MPS cần thời gian để khởi tạo bộ nhớ, cache đồ họa và nạp nhân (kernels) trong lần chạy đầu tiên, bắt buộc phải chạy tối thiểu **50 lượt suy diễn giả định (warm-up runs)** trước khi bắt đầu bấm giờ đo thực tế.
3. **Đồng bộ hóa luồng GPU (Device Synchronization)**:
   * Vì GPU thực thi lệnh bất đồng bộ (asynchronous execution), nếu không đồng bộ luồng trước khi đo thời gian, phép đo sẽ bị sai lệch (chỉ đo được thời gian đẩy lệnh vào queue thay vì thời gian GPU tính toán xong).
   * Bắt buộc dùng `torch.cuda.synchronize()` (cho CUDA) hoặc `torch.mps.synchronize()` (cho Apple Silicon) ngay trước và sau khi thực hiện đo thời gian suy diễn.

---

## 2. Các chỉ số cần báo cáo

Agent phải thu thập thời gian suy diễn của tối thiểu **100 mẫu kiểm thử liên tiếp** và báo cáo các chỉ số (đơn vị: mili-giây - ms):
* **Mean Latency**: Thời gian suy diễn trung bình.
* **p50 Latency (Median)**: Thời gian suy diễn ở phân vị 50%.
* **p95 Latency**: Thời gian suy diễn ở phân vị 95% (độ trễ tối đa mà 95% lượng ảnh đạt được).
* **p99 Latency**: Thời gian suy diễn ở phân vị 99% (đánh giá các trường hợp trễ đột biến - latency spikes).

---

## 3. Minh họa code đo Latency trong PyTorch

```python
import time
import torch

def measure_latency(model, sample_input, device, num_runs=100, warm_up=50):
    model.to(device)
    model.eval()
    
    # Chuẩn bị dữ liệu
    sample_input = sample_input.to(device)
    
    # Warm-up phase
    with torch.no_grad():
        for _ in range(warm_up):
            _ = model(sample_input)
            
    # Đồng bộ hóa thiết bị sau khi warm-up
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
        
    latencies = []
    
    # Đo thực tế
    with torch.no_grad():
        for _ in range(num_runs):
            t_start = time.perf_counter()
            _ = model(sample_input)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()
                
            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1000) # Quy đổi sang ms
            
    latencies = np.array(latencies)
    
    return {
        "mean": np.mean(latencies),
        "p50": np.percentile(latencies, 50),
        "p95": np.percentile(latencies, 95),
        "p99": np.percentile(latencies, 99)
    }
```
