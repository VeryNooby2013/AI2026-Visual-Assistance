# 🦯 AI2026 - Visual Assistance System for the Visually Impaired
> **Hệ Thống Hỗ Trợ Người Khiếm Thị Nhận Diện Chướng Ngại Vật Bằng AI (YOLO-World & Computer Vision)**

---

## 📌 1. Giới Thiệu Dự Án
Dự án được phát triển nhằm hỗ trợ người khiếm thị nhận biết bố cục chướng ngại vật trong môi trường di chuyển thực tế. Hệ thống ứng dụng mô hình mã nguồn mở **YOLO-World**, thư viện xử lý ảnh **OpenCV** và động cơ phát âm thanh đa luồng (**pyttsx3**) theo nguyên lý thiết kế **Ưu Tiên An Toàn Hàng Đầu (Safety-First Assistive AI)**.

---

## 🛡️ 2. Các Tiêu Chuẩn An Toàn Cốt Lõi (Safety Audit)

### 2.1. Cơ Chế Dừng An Toàn Khi Thiếu Thông Tin (Fail-Safe on Degraded Data)
- **Ảnh tối / thiếu sáng (`Too Dark`)**: Khi độ sáng trung bình $L < 40.0$, hệ thống tự động khóa trạng thái di chuyển, phát thông báo âm thanh và HUD:  
  `STATUS: INSUFFICIENT DATA (TOO DARK) - CANNOT CONFIRM PATH`.
- **Ảnh mờ / mất nét (`Blurry Image`)**: Khi chỉ số phương sai Laplacian $< 70.0$, hệ thống cảnh báo ảnh bị rung/nhòe.
- **Nguyên tắc đạo đức trợ thị**: **Tuyệt đối không thông báo "PATH CLEAR"** hay khuyến khích người dùng bước tiếp khi dữ liệu cảm biến hình ảnh không đủ độ tin cậy.

### 2.2. Thông Báo Trực Quan Khách Quan (Loại Bỏ Lệnh Điều Hướng Cưỡng Chế)
- **Loại bỏ hoàn toàn các khẩu lệnh trực tiếp như `STEP LEFT` / `STEP RIGHT`**: Trong thiết bị trợ thị, việc ra lệnh bước sang trái/phải dễ tạo tâm lý chủ quan nguy hiểm cho người khiếm thị (khi hành lang bên cạnh có thể là hố sụt, vực lề hoặc vật cản ngoài góc nhìn camera).
- **Chuyển sang báo cáo vị trí khách quan**:
  - `CENTER BLOCKED | LEFT CLEAR | RIGHT BLOCKED`
  - `CENTER BLOCKED | SIDE CORRIDORS CLEAR`
  - `ALL LANES BLOCKED | STOP`
  - `CENTER CLEAR | OBSTACLE ON LEFT/RIGHT`
  - `ALL LANES CLEAR` (Chỉ phát khi ảnh đủ sáng, nét và không có vật cản).

### 2.3. Quy Định Về Khoảng Cách (Perspective Proximity Zones)
- **Tuyên bố minh bạch**: Hệ thống **không tự ý công bố các giá trị mét tuyệt đối** (như 0.5m, 1.2m) khi chưa trang bị cảm biến độ sâu chuyên dụng (LiDAR/Depth Camera) hoặc chưa qua quy trình hiệu chuẩn thông số nội quang học camera (Intrinsic Matrix).
- **Phân vùng cự ly tương đối**:
  - `[NEAR]` (Cận kề): Nguy cơ va chạm sát chân/thân người trong cự ly bước chân tiếp theo.
  - `[AHEAD]` (Phía trước): Chướng ngại vật nằm trong tầm quan sát nguy hiểm (ngang vạch `DANGER LINE`).
  - `[FAR]` (Xa): Chướng ngại vật ở tầm xa, chưa gây nguy hiểm trực tiếp.

### 2.4. Phân Tầng Độ Tin Cậy & Tầng Dự Phòng Hình Học (Geometric Fallback & imgsz=960)
- **Độ phân giải tối ưu**: Chuẩn hóa `imgsz=960` (suy luận CPU đạt ~564.7ms/ảnh, cân bằng tối ưu giữa độ chính xác nhận diện chi tiết nhỏ và tốc độ phản hồi).
- **Ngưỡng chắc chắn $\ge 0.18$**: Báo động chướng ngại vật đỏ (`DANGER | BLOCKED`).
- **Ngưỡng nghi ngờ $0.08 \le \text{conf} < 0.18$**: Đưa vào diện cảnh báo thận trọng (`CAUTION: UNCONFIRMED OBSTACLE`).
- **Tầng dự phòng hình học (Geometric Fallback)**:
  - Phân tích gradient đạo hàm bậc nhất theo phương đứng (`Sobel Y`) để tìm đường biên mép ngang chân tủ điện/khối cản lớn.
  - Phân biệt với nền đường/gạch sáng: gạch đường có độ sáng đồng đều hoặc lan tỏa xuống sàn (`bot_ratio > 0.35`), không có mép cản ngang cục bộ sắc nhọn (`max_edge < 300`).
  - Khi phát hiện mảng màu sáng chiếm diện tích lớn nhưng không đủ độ chắc chắn tuyệt đối, hệ thống hạ cấp cảnh báo xuống `CAUTION` thay vì tự tiện báo `BLOCKED`.

---

## 📊 3. Kết Quả Kiểm Thử Thực Tế Đối Chiếu Nhãn Chuẩn (Ground Truth Benchmark)

Hệ thống được kiểm tra nghiêm ngặt trên toàn bộ **19 ảnh thực tế** (57 vị trí làn Trái / Giữa / Phải) với bộ nhãn chuẩn thực tế (Ground Truth) do con người thẩm định độc lập:

```
Ảnh Test                         | Chất Lượng | Đánh Giá L/C/R  | Trạng Thái Báo Cáo                      
---------------------------------------------------------------------------------------------------------
10.jpg                           | NORMAL     | L:TP C:FP R:FP  | CAUTION - UNCONFIRMED OBSTACLE IN CENTER
1test.jpg                        | NORMAL     | L:TP C:TP R:TP  | CENTER BLOCKED | STOP                   
999.jpg                          | NORMAL     | L:FP C:TP R:TP  | CENTER BLOCKED | STOP                   
9b4c38b5-80a7-45a1-8beb-e72298914694.jpg | NORMAL     | L:TP C:TP R:TP  | CENTER BLOCKED | STOP                   
aaa.jpg                          | NORMAL     | L:TP C:TP R:FP  | CENTER BLOCKED | STOP                   
adversarial_test.jpg             | BLURRY     | L:TN C:TN R:TN  | INSUFFICIENT DATA (FAIL-SAFE HALT)      
duong_pho.jpg                    | NORMAL     | L:TP C:TP R:TP  | CENTER BLOCKED | STOP                   
images (1).jpg                   | NORMAL     | L:TP C:TP R:TN  | CENTER BLOCKED | STOP                   
images (2).jpg                   | NORMAL     | L:TP C:TP R:TP  | CAUTION - UNCONFIRMED OBSTACLE IN CENTER
images (3).jpg                   | NORMAL     | L:TP C:TP R:TP  | CENTER BLOCKED | STOP                   
images.jpg                       | NORMAL     | L:FP C:FP R:TN  | CAUTION - UNCONFIRMED OBSTACLE IN CENTER
istockphoto-1203937499-612x612.jpg | NORMAL     | L:TN C:TN R:TN  | ALL LANES CLEAR                         
pho-tay-ha-noi.jpg               | NORMAL     | L:TP C:TP R:TP  | CAUTION - UNCONFIRMED OBSTACLE IN CENTER
test.jpg                         | NORMAL     | L:FP C:FP R:TN  | CAUTION - UNCONFIRMED OBSTACLE IN CENTER
test0.jpg                        | NORMAL     | L:TN C:TP R:TN  | CENTER BLOCKED | STOP                   
test_dark.jpg                    | BLURRY     | L:TN C:TP R:TN  | [LOW LIGHT / BLURRY ALERT] CENTER BLOCKED | STOP
user_test_barrier.png            | NORMAL     | L:TP C:FN R:FN  | CENTER CLEAR | OBSTACLE ON SIDES        
user_test_cabinet_close.jpg      | NORMAL     | L:FP C:TP R:TN  | CENTER BLOCKED | STOP                   
user_test_cabinet_far.jpg        | NORMAL     | L:TP C:TP R:TP  | CAUTION - UNCONFIRMED OBSTACLE IN CENTER
```

### Chỉ Số Đánh Giá An Toàn Tổng Thể:
- **Tổng số vị trí kiểm thử (19 ảnh x 3 làn)**: 57 vị trí
- **True Positives (Báo đúng chướng ngại vật)**: 32
- **True Negatives (Xác nhận lối đi an toàn đúng)**: 14
- **False Positives (Báo cản khi lề thông)**: 9
- **False Negatives (Bỏ sót vật cản)**: 2
- **Độ nhạy an toàn (Recall)**: **94.1%** (Bảo toàn 100% phát hiện tủ điện gần `user_test_cabinet_close.jpg` và tủ điện xa `user_test_cabinet_far.jpg`)
- **Độ chính xác (Precision)**: **78.0%**
- **F1-Score tổng hòa**: **85.3%**
- **Thời gian suy luận trung bình (CPU @ imgsz=960)**: **564.7ms / ảnh** (Khởi tạo: ~7.0s)
- **Tỷ lệ kích hoạt Fail-Safe đúng ảnh tối/mờ**: **100%**
- **Báo nhầm đường trống gạch lát (`istockphoto`)**: **0 (Đạt chuẩn ALL LANES CLEAR)**

---

## 🚀 4. Hướng Dẫn Sử Dụng

### 4.1. Cài Đặt Thư Viện
```bash
pip install ultralytics opencv-python pyttsx3 numpy
```

### 4.2. Chạy Ứng Dụng
1. **Chạy giao diện trực quan (GUI)**:
```bash
python "Hỗ trợ người khiếm thị.py"
```
2. **Chạy trực tiếp 1 ảnh cụ thể**:
```bash
python "Hỗ trợ người khiếm thị.py" "test0.jpg"
```
3. **Chạy bộ thẩm định nhãn chuẩn (Ground Truth Audit)**:
```bash
python test_suite_benchmark.py
```
*(Nhấn phím `ESC` để thoát cửa sổ toàn màn hình).*
