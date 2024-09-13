import cv2
from Yolo_v5 import YOLOv5

# Đường dẫn đến mô hình YOLOv5 của bạn
model_path = r'E:\Personal_Project\Hệ thống giám sát giao thông\model\yolov5s.pt'

# Tạo đối tượng mô hình YOLOv5 từ tệp .pt
model = YOLOv5(model_path)

# Khởi động webcam
cap = cv2.VideoCapture(0)

while True:
    # Đọc khung hình từ webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển đổi khung hình từ BGR sang RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Dự đoán với mô hình YOLOv5
    results = model.predict(img_rgb)

    # Hiển thị kết quả
    results.render()  # Thêm bounding boxes vào khung hình
    frame_with_boxes = results.ims[0]

    # Hiển thị khung hình có bounding boxes
    cv2.imshow('YOLOv5 Webcam', frame_with_boxes)

    # Thoát khi nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
