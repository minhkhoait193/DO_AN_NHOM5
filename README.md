# 🚗 Hệ thống Nhận diện và Theo dõi Biển số xe bằng YOLOv8 + SORT

## 🧠 Mô tả

Dự án này sử dụng mô hình YOLOv8 để phát hiện xe và biển số từ video đầu vào, sau đó sử dụng thuật toán SORT để theo dõi xe. Biển số sẽ được đọc bằng EasyOCR. Dữ liệu được xuất ra file `.csv`, sau đó được nội suy để điền đầy đủ thông tin khung hình, cuối cùng tạo ra video hiển thị kết quả mượt mà.

---

## 📂 Cấu trúc thư mục
├── main.py # Nhận diện xe & biển số, xuất test.csv
├── add_missing_data.py # Nội suy dữ liệu còn thiếu -> test_interpolated.csv
├── visualize.py # Vẽ bounding box + hiển thị biển số lên video
├── util.py # Các hàm hỗ trợ: OCR, định dạng biển số, xuất csv
├── requirements.txt # Thư viện cần thiết
├── sample.mp4 # Video đầu vào
├── yolov8n.pt # Mô hình YOLOv8 phát hiện xe
├── license_plate_detector.pt # Mô hình YOLOv8 phát hiện biển số
├── test.csv # Kết quả thô từ YOLO + OCR
├── test_interpolated.csv # Kết quả sau nội suy
└── sort/
└── sort.py # Thuật toán SORT (tracking)


---

## 📥 Tải dữ liệu & mô hình

- 🔗 Video mẫu: [Download tại đây](https://drive.google.com/file/d/1JbwLyqpFCXmftaJY1oap8Sa6KfjoWJta/view?usp=sharing)
- 🔗 Mô hình biển số: [Download tại đây](https://drive.google.com/file/d/1Zmf5ynaTFhmln2z7Qvv-tgjkWQYQ9Zdw/view?usp=sharing)
- 🔗 SORT gốc: [GitHub SORT](https://github.com/abewley/sort) *(đã được đưa vào thư mục `sort/`)*

---

## 🧪 Cài đặt môi trường

> ⚠️ Yêu cầu: Python 3.10, pip, pip install hỗ trợ GPU nếu có thể.

1. **Tạo môi trường (nếu dùng conda)**:
```bash
conda create --prefix ./env python=3.10 -y
conda activate ./env
```

Cài đặt thư viện cần thiết:
pip install -r requirements.txt

Nếu thiếu lap hoặc scikit-image, bạn có thể thêm:
pip install lap scikit-image

Cách sử dụng
```bash



