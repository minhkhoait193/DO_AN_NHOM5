from ultralytics import YOLO                 # Import YOLOv8 để phát hiện xe và biển số
import cv2                                   # OpenCV: xử lý ảnh và video
import util                                  # Import module util chứa các hàm hỗ trợ như đọc biển số, ghi csv
from sort.sort import *                      # Import thuật toán SORT để theo dõi xe giữa các khung hình
from util import get_car, read_license_plate, write_csv  # Hàm hỗ trợ xử lý biển số và ghi dữ liệu

# Khởi tạo biến lưu kết quả và tracker
results = {}                                 # Dictionary lưu kết quả theo từng khung hình
mot_tracker = Sort()                         # Khởi tạo đối tượng SORT để theo dõi xe

# Load mô hình phát hiện
coco_model = YOLO('yolov8n.pt')              # Mô hình YOLOv8 dùng để phát hiện xe (class như car, bus, truck)
license_plate_detector = YOLO('license_plate_detector.pt')  # Mô hình YOLO riêng để phát hiện biển số xe

# Mở video đầu vào
cap = cv2.VideoCapture('./sample.mp4')       # Đối tượng đọc video
vehicles = [2, 3, 5, 7]                       # ID lớp trong YOLO tương ứng với các loại xe (car, motorbike, bus, truck)

# Đọc từng khung hình và xử lý
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()

    # Giới hạn chỉ xử lý 1000 khung hình đầu (nếu muốn xử lý toàn bộ thì bỏ điều kiện frame_nmr < 1000)
    if ret and frame_nmr < 1000:
        results[frame_nmr] = {}

        # -------------------------
        # 1. Phát hiện xe bằng YOLO
        # -------------------------
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # -----------------------------
        # 2. Theo dõi xe bằng SORT
        # -----------------------------
        track_ids = mot_tracker.update(np.asarray(detections_))  # Trả về tọa độ xe + ID được gán

        # ----------------------------------------
        # 3. Phát hiện biển số bằng mô hình riêng
        # ----------------------------------------
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # -------------------------------------
            # 4. Gán biển số vào xe phù hợp (nếu có)
            # -------------------------------------
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
            if car_id != -1:

                # -------------------------------
                # 5. Cắt và xử lý vùng biển số
                # -------------------------------
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # -----------------------------------
                # 6. Đọc ký tự trên biển số bằng OCR
                # -----------------------------------
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                # ------------------------------------
                # 7. Nếu đọc thành công thì lưu kết quả
                # ------------------------------------
                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'text': license_plate_text,
                            'bbox_score': score,
                            'text_score': license_plate_text_score
                        }
                    }

# ---------------------------------------------
# 8. Sau khi xử lý xong tất cả các frame, ghi ra CSV
# ---------------------------------------------
write_csv(results, './test.csv')
