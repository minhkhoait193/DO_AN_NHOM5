# 🛰️ SORT (Simple Online and Realtime Tracking)

Đây là mã nguồn của **SORT** – một thuật toán theo dõi đối tượng theo thời gian thực đơn giản và hiệu quả, được sử dụng để theo dõi các vật thể (ví dụ: xe, người) trong video.

> 📌 SORT đã được tích hợp vào project chính để theo dõi xe dựa trên bounding box từ YOLOv8.

---

## 📘 Giới thiệu

SORT hoạt động bằng cách kết hợp:
- **Kalman Filter** – dự đoán vị trí tiếp theo của đối tượng.
- **Hungarian Algorithm** – để kết nối (match) các bounding box giữa khung hình hiện tại và trước đó dựa trên IOU (Intersection over Union).

Đây là phiên bản gốc do [Alex Bewley](https://github.com/abewley/sort) phát triển, đã được đóng gói lại vào thư mục `soft` này để dùng trực tiếp trong project chính.

---

## 📦 Cấu trúc

soft/
├── sort.py # Toàn bộ mã nguồn của thuật toán SORT

├── requirements.txt # Thư viện cần thiết để chạy SORT

└── README.md # (File này)

---

## 📥 Cài đặt thư viện cần thiết

Chạy lệnh sau để cài các thư viện:

```bash
pip install -r requirements.txt
```

Nếu không dùng requirements.txt, bạn có thể cài trực tiếp:
```bash
pip install filterpy lap scikit-image
```
Cách sử dụng cơ bản
---

```bash
from sort import Sort

mot_tracker = Sort()  # Tạo đối tượng tracker

# giả sử bạn có bounding boxes theo dạng [x1, y1, x2, y2, confidence]
detections = [[100, 150, 200, 300, 0.9], [220, 160, 280, 310, 0.8]]

# Gọi update mỗi khung hình
tracked_objects = mot_tracker.update(np.array(detections))

```
Output (tracked_objects) sẽ là array với thông tin:
```bash
[[x1, y1, x2, y2, ID],
 [x1, y1, x2, y2, ID],
 ...]
```


