# Thư viện đọc và ghi file CSV
import csv
# Thư viện xử lý mảng số, dùng để tính toán vector
import numpy as np
# Hàm nội suy tuyến tính để tính giá trị giữa 2 điểm
from scipy.interpolate import interp1d



# Hàm chính để nội suy khung xe và biển số bị thiếu trong các khung hình
def interpolate_bounding_boxes(data):
    # Extract necessary data columns from input data
# Lấy danh sách số khung hình từ dữ liệu
    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
# Lấy danh sách ID của các xe
    car_ids = np.array([int(float(row['car_id'])) for row in data])
# Lấy tọa độ khung xe
    car_bboxes = np.array([list(map(float, row['car_bbox'][1:-1].split())) for row in data])
# Lấy tọa độ khung biển số xe
    license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'][1:-1].split())) for row in data])

    interpolated_data = []
# Lấy danh sách ID của các xe
    unique_car_ids = np.unique(car_ids)
# Lấy danh sách các ID xe không trùng lặp
    for car_id in unique_car_ids:

# Lấy danh sách frame ban đầu có mặt xe này
        frame_numbers_ = [p['frame_nmr'] for p in data if int(float(p['car_id'])) == int(float(car_id))]
        print(frame_numbers_, car_id)

        # Filter data for a specific car ID
# Tạo mask lọc ra dòng nào thuộc xe hiện tại
        car_mask = car_ids == car_id
# Lấy danh sách số khung hình từ dữ liệu
        car_frame_numbers = frame_numbers[car_mask]
        car_bboxes_interpolated = []
        license_plate_bboxes_interpolated = []

        first_frame_number = car_frame_numbers[0]
        last_frame_number = car_frame_numbers[-1]

# Lặp qua từng frame mà xe xuất hiện để kiểm tra khoảng cách giữa các frame
        for i in range(len(car_bboxes[car_mask])):
            frame_number = car_frame_numbers[i]
            car_bbox = car_bboxes[car_mask][i]
            license_plate_bbox = license_plate_bboxes[car_mask][i]

            if i > 0:
                prev_frame_number = car_frame_numbers[i-1]
                prev_car_bbox = car_bboxes_interpolated[-1]
                prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]

# Nếu có khoảng trống giữa 2 frame → thực hiện nội suy
                if frame_number - prev_frame_number > 1:
                    # Interpolate missing frames' bounding boxes
                    frames_gap = frame_number - prev_frame_number
                    x = np.array([prev_frame_number, frame_number])
                    x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
# Tạo hàm nội suy tuyến tính cho khung hình hoặc biển số
                    interp_func = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind='linear')
# Lấy tọa độ khung xe
                    interpolated_car_bboxes = interp_func(x_new)
# Tạo hàm nội suy tuyến tính cho khung hình hoặc biển số
                    interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0, kind='linear')
# Lấy tọa độ khung biển số xe
                    interpolated_license_plate_bboxes = interp_func(x_new)

                    car_bboxes_interpolated.extend(interpolated_car_bboxes[1:])
                    license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])

# Thêm bounding box vào danh sách kết quả đã nội suy
            car_bboxes_interpolated.append(car_bbox)
            license_plate_bboxes_interpolated.append(license_plate_bbox)


# Tạo dòng dữ liệu mới cho mỗi frame sau nội suy
        for i in range(len(car_bboxes_interpolated)):
            frame_number = first_frame_number + i
            row = {}
            row['frame_nmr'] = str(frame_number)
            row['car_id'] = str(car_id)
            row['car_bbox'] = ' '.join(map(str, car_bboxes_interpolated[i]))
            row['license_plate_bbox'] = ' '.join(map(str, license_plate_bboxes_interpolated[i]))

# Nếu là frame được nội suy → set điểm số và nội dung = 0
            if str(frame_number) not in frame_numbers_:
                # Imputed row, set the following fields to '0'
                row['license_plate_bbox_score'] = '0'
                row['license_number'] = '0'
                row['license_number_score'] = '0'
            else:
                # Original row, retrieve values from the input data if available
                original_row = [p for p in data if int(p['frame_nmr']) == frame_number and int(float(p['car_id'])) == int(float(car_id))][0]
                row['license_plate_bbox_score'] = original_row['license_plate_bbox_score'] if 'license_plate_bbox_score' in original_row else '0'
                row['license_number'] = original_row['license_number'] if 'license_number' in original_row else '0'
                row['license_number_score'] = original_row['license_number_score'] if 'license_number_score' in original_row else '0'

# Thêm dòng hoàn chỉnh vào danh sách kết quả
            interpolated_data.append(row)

    return interpolated_data


# Load the CSV file

# Mở file test.csv để đọc dữ liệu gốc
with open('test.csv', 'r') as file:
    reader = csv.DictReader(file)
    data = list(reader)

# Interpolate missing data
# Gọi hàm nội suy để xử lý toàn bộ dữ liệu
interpolated_data = interpolate_bounding_boxes(data)

# Write updated data to a new CSV file
header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']

# Ghi dữ liệu đã nội suy vào file mới test_interpolated.csv
with open('test_interpolated.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    writer.writerows(interpolated_data)
