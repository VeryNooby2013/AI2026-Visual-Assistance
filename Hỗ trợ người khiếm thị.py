import cv2
import threading
import pyttsx3
import tkinter as tk
from tkinter import filedialog
import numpy as np
import sys
import os
from ultralytics import YOLOWorld

# ==========================================
# 1. AUDIO ALERT ENGINE (ENGLISH TTS)
# ==========================================
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 160)

def speak_async(text):
    def _run():
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True).start()

# ==========================================
# 2. SELECT TEST IMAGE FILE (GUI / CLI)
# ==========================================
image_path = None
if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
    image_path = sys.argv[1]
else:
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    image_path = filedialog.askopenfilename(
        title="Select Test Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp *.bmp")]
    )

if not image_path:
    exit()

frame = cv2.imread(image_path)
if frame is None:
    exit()

h, w, _ = frame.shape
one_third = w // 3
two_thirds = 2 * (w // 3)
danger_y = int(h * 0.46)

# ==========================================
# 3. IMAGE QUALITY CHECKS (DARK / BLURRY)
# ==========================================
gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
brightness = float(gray_full.mean())
blur_score = float(cv2.Laplacian(gray_full, cv2.CV_64F).var())

is_dark = brightness < 40.0
is_overexposed = brightness > 230.0
is_blurry = blur_score < 70.0
is_quality_degraded = is_dark or is_overexposed or is_blurry

quality_warning = None
if is_dark:
    quality_warning = f"INSUFFICIENT DATA: TOO DARK (Bri {brightness:.1f})"
elif is_overexposed:
    quality_warning = f"INSUFFICIENT DATA: OVEREXPOSED (Bri {brightness:.1f})"
elif is_blurry:
    quality_warning = f"INSUFFICIENT DATA: BLURRY (Focus {blur_score:.1f})"

# ==========================================
# 4. INITIALIZE YOLO-WORLD MODEL & THRESHOLDS
# ==========================================
model = YOLOWorld('yolov8s-world.pt')

CUSTOM_CLASSES = [
    "cabinet", "electric box", "obstacle", "barrier", "fence",
    "box", "barrel", "sign", "person", "vehicle", "wall", "pillar"
]
model.set_classes(CUSTOM_CLASSES)

CONF_CONFIRMED = 0.18   # Ngưỡng vật cản rõ ràng
CONF_UNCERTAIN = 0.08   # Ngưỡng nghi ngờ / lọc nhiễu dưới 0.08

def get_proximity_zone(y2, h_frame, danger_line_y):
    """Phân loại vùng cự ly tương đối theo phối cảnh mặt đất."""
    if y2 < danger_line_y:
        return "FAR"
    elif y2 >= danger_line_y + int((h_frame - danger_line_y) * 0.45):
        return "NEAR"
    else:
        return "AHEAD"

# Chạy mô hình suy luận (chốt imgsz=960 theo benchmark CPU tối ưu tốc độ và độ chính xác)
results = model.predict(frame, conf=CONF_UNCERTAIN, iou=0.40, imgsz=960, verbose=False)

left_blocked = False
center_blocked = False
right_blocked = False

left_caution = False
center_caution = False
right_caution = False

closest_center_zone = None
closest_left_zone = None
closest_right_zone = None

# ==========================================
# 5. 3-LANE SPATIAL ANALYSIS (LOGIC LẤN LÀN >= 10%)
# ==========================================
if results[0].boxes is not None and len(results[0].boxes) > 0:
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    confs = results[0].boxes.conf.cpu().numpy()
    clss = results[0].boxes.cls.cpu().numpy().astype(int)

    for box, conf, cls_id in zip(boxes, confs, clss):
        x1, y1, x2, y2 = box
        box_w = max(x2 - x1, 1)
        box_h = max(y2 - y1, 1)
        area = box_w * box_h

        # Lọc bỏ nhiễu trên cao hoặc vụn vặt
        if y2 < int(h * 0.20) or area < (w * h * 0.002):
            continue

        cls_name = CUSTOM_CLASSES[cls_id] if cls_id < len(CUSTOM_CLASSES) else "obstacle"
        is_confirmed = (conf >= CONF_CONFIRMED)
        zone = get_proximity_zone(y2, h, danger_y)
        cx = (x1 + x2) // 2

        # Tính độ bao phủ của vật thể trên làn giữa [one_third, two_thirds]
        overlap_left = max(x1, one_third)
        overlap_right = min(x2, two_thirds)
        mid_overlap = max(0, overlap_right - overlap_left)
        overlap_ratio = mid_overlap / box_w

        # Logic chuẩn: Vật cản lấn vào giữa >= 10% bề rộng hoặc nằm trong làn giữa
        in_center = (overlap_ratio >= 0.10) or (one_third <= cx <= two_thirds)
        
        # Vật cản nguy hiểm khi chân vượt qua DANGER LINE hoặc là vật thể lớn chiếm tầm nhìn
        is_danger = (y2 >= danger_y) or (in_center and box_h > int(h * 0.30))

        # Tính độ bao phủ 2 làn bên
        left_overlap = max(0, min(x2, one_third) - max(x1, 0))
        right_overlap = max(0, min(x2, w) - max(x1, two_thirds))
        in_left = (left_overlap / box_w >= 0.15) or (cx < one_third)
        in_right = (right_overlap / box_w >= 0.15) or (cx > two_thirds)

        if is_danger:
            if in_center:
                if is_confirmed:
                    center_blocked = True
                    if closest_center_zone != "NEAR": closest_center_zone = zone
                else:
                    center_caution = True
                    if not closest_center_zone: closest_center_zone = zone
            if in_left:
                if is_confirmed:
                    left_blocked = True
                    if closest_left_zone != "NEAR": closest_left_zone = zone
                else:
                    left_caution = True
            if in_right:
                if is_confirmed:
                    right_blocked = True
                    if closest_right_zone != "NEAR": closest_right_zone = zone
                else:
                    right_caution = True

            # Vẽ bounding box trực quan
            if is_confirmed:
                box_color = (0, 0, 255)  # Đỏ: Vật cản chắc chắn
                label_text = f"{cls_name.upper()} [{zone}] ({int(conf*100)}%)"
            else:
                box_color = (0, 165, 255)  # Cam: AI nghi ngờ
                label_text = f"[UNCERTAIN] {cls_name} ({int(conf*100)}%)"

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, label_text, (x1, max(y1 - 8, 45)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, box_color, 2)

# ==========================================
# 6. TẦNG DỰ PHÒNG HÌNH HỌC: TỦ ĐIỆN / KHỐI CẢN LỚN MÀU SÁNG
# Phân biệt nền đường/gạch sáng với vật cản bằng bằng chứng hình học (Sobel Y boundary)
# ==========================================
if not center_blocked:
    # Quét dải màu sáng/trắng ở làn giữa từ tầm mắt xuống sát chân
    mid_roi = frame[int(h * 0.10):int(h * 0.85), one_third:two_thirds]
    if mid_roi.size > 0:
        gray_roi = cv2.cvtColor(mid_roi, cv2.COLOR_BGR2GRAY)
        roi_h, roi_w = mid_roi.shape[:2]
        _, white_mask = cv2.threshold(gray_roi, 170, 255, cv2.THRESH_BINARY)
        white_pixels = cv2.countNonZero(white_mask)
        roi_area = roi_h * roi_w
        white_ratio = white_pixels / roi_area

        # Bằng chứng hình học: Tủ điện có gờ cạnh ngang phân cách sắc nét (chân tủ, khe viền)
        # với gradient Sobel Y cục bộ cực mạnh, khác biệt hoàn toàn nền đường/gạch phẳng.
        sob_y = np.abs(cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3))
        row_means = sob_y.mean(axis=1)
        max_edge = float(row_means.max())
        max_edge_row = float(row_means.argmax() / roi_h)

        abs_x1 = one_third + 15
        abs_y1 = int(h * 0.10)
        abs_x2 = two_thirds - 15
        abs_y2 = int(h * 0.72)

        # Trường hợp 1: Vật cản chắc chắn (Tủ điện đứng, có mép ngang rõ rệt ở tầm thân/chân)
        if white_ratio > 0.28 and max_edge >= 450.0 and (0.35 <= max_edge_row <= 0.85):
            box_color = (0, 0, 255)
            cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), box_color, 2)
            cv2.putText(frame, "[DANGER: CENTER] (CABINET / OBSTACLE)", (abs_x1, max(abs_y1 - 10, 40)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)
            center_blocked = True
            closest_center_zone = "NEAR" if abs_y2 >= danger_y else "AHEAD"
        # Trường hợp 2: Không đủ bằng chứng chắc chắn nhưng có dấu hiệu bất thường -> Báo CAUTION thay vì BLOCKED
        elif white_ratio > 0.35 and max_edge >= 300.0:
            box_color = (0, 165, 255)
            cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), box_color, 2)
            cv2.putText(frame, "[CAUTION: CENTER] (UNCONFIRMED BLOCK)", (abs_x1, max(abs_y1 - 10, 40)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)
            center_caution = True
            if not closest_center_zone: closest_center_zone = "AHEAD"

# ==========================================
# 7. STATUS & AUDIO ALERTS
# (ƯU TIÊN CẢNH BÁO VẬT CẢN KỂ CẢ KHI ẢNH TỐI/MỜ)
# ==========================================
speech_text = ""
status_text = ""
status_color = (0, 255, 0)
prefix = "[LOW LIGHT / BLURRY ALERT] " if is_quality_degraded else ""

# TRƯỜNG HỢP 1: LÀN GIỮA BỊ CHẶN (CENTER BLOCKED)
if center_blocked:
    status_color = (0, 0, 255)
    c_zone_str = f" [{closest_center_zone}]" if closest_center_zone else ""
    
    if not left_blocked and right_blocked:
        status_text = f"STATUS: {prefix}CENTER BLOCKED{c_zone_str} | LEFT CLEAR | RIGHT BLOCKED"
        speech_text = "Warning, obstacle detected in center. Left corridor appears clear."
    elif not right_blocked and left_blocked:
        status_text = f"STATUS: {prefix}CENTER BLOCKED{c_zone_str} | LEFT BLOCKED | RIGHT CLEAR"
        speech_text = "Warning, obstacle detected in center. Right corridor appears clear."
    elif not left_blocked and not right_blocked:
        status_text = f"STATUS: {prefix}CENTER BLOCKED{c_zone_str} | SIDE CORRIDORS CLEAR"
        speech_text = "Warning, obstacle detected in center. Side corridors appear clear."
    else:
        status_text = f"STATUS: {prefix}ALL LANES BLOCKED{c_zone_str} | STOP"
        speech_text = "Warning, path is fully blocked. Stop."
        
    if is_quality_degraded:
        speech_text = ("Warning, low light. " if is_dark else "Warning, blurry image. ") + speech_text

# TRƯỜNG HỢP 2: AI NGHI NGỜ Ở LÀN GIỮA (UNCERTAIN OBSTACLE)
elif center_caution:
    status_color = (0, 165, 255)
    status_text = f"STATUS: {prefix}CAUTION - UNCONFIRMED OBSTACLE IN CENTER"
    speech_text = "Caution, possible obstacle ahead in center. Proceed with care."
    if is_quality_degraded:
        speech_text = ("Warning, low light. " if is_dark else "Warning, blurry image. ") + speech_text

# TRƯỜNG HỢP 3: LÀN GIỮA THÔNG THOÁNG, VẬT CẢN Ở HAI BÊN
elif left_blocked or left_caution or right_blocked or right_caution:
    status_color = (0, 220, 220)
    if (left_blocked or left_caution) and (right_blocked or right_caution):
        status_text = f"STATUS: {prefix}CENTER CLEAR | OBSTACLES ON BOTH SIDES"
        speech_text = "Center path is clear. Obstacles detected on both sides."
    elif left_blocked or left_caution:
        status_text = f"STATUS: {prefix}CENTER CLEAR | OBSTACLE ON LEFT"
        speech_text = "Center path is clear. Obstacle detected on left side."
    else:
        status_text = f"STATUS: {prefix}CENTER CLEAR | OBSTACLE ON RIGHT"
        speech_text = "Center path is clear. Obstacle detected on right side."
    if is_quality_degraded:
        speech_text = ("Warning, low light. " if is_dark else "Warning, blurry image. ") + speech_text

# TRƯỜNG HỢP 4: KHÔNG PHÁT HIỆN VẬT CẢN
else:
    if is_quality_degraded:
        status_color = (0, 165, 255)
        if is_dark:
            status_text = "STATUS: INSUFFICIENT DATA (TOO DARK) - CANNOT CONFIRM PATH"
            speech_text = "Warning, image is too dark. Insufficient data, cannot confirm clear path."
        elif is_blurry:
            status_text = "STATUS: INSUFFICIENT DATA (BLURRY) - CANNOT CONFIRM PATH"
            speech_text = "Warning, image is blurry. Insufficient data, cannot confirm clear path."
        else:
            status_text = "STATUS: INSUFFICIENT DATA (GLARE) - CANNOT CONFIRM PATH"
            speech_text = "Warning, severe glare. Insufficient data, cannot confirm clear path."
    else:
        status_color = (0, 255, 0)
        status_text = "STATUS: ALL LANES CLEAR"
        speech_text = "All lanes clear."

speak_async(speech_text)

# ==========================================
# 8. MINIMAL HUD (LANES, DANGER LINE & STATUS)
# ==========================================
# Vạch phân định 3 làn
cv2.line(frame, (one_third, 0), (one_third, h), (180, 180, 180), 1)
cv2.line(frame, (two_thirds, 0), (two_thirds, h), (180, 180, 180), 1)

def get_lane_ui(blocked, caution, zone):
    if blocked:
        z_str = f" [{zone}]" if zone else ""
        return f"BLOCKED{z_str}", (0, 0, 255)
    if caution:
        return "CAUTION", (0, 165, 255)
    return "CLEAR", (0, 255, 0)

l_str, l_col = get_lane_ui(left_blocked, left_caution, closest_left_zone)
c_str, c_col = get_lane_ui(center_blocked, center_caution, closest_center_zone)
r_str, r_col = get_lane_ui(right_blocked, right_caution, closest_right_zone)

# Trạng thái từng làn ở góc trên
cv2.putText(frame, f"LEFT: {l_str}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.60, l_col, 2)
cv2.putText(frame, f"CENTER: {c_str}", (one_third + 15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.60, c_col, 2)
cv2.putText(frame, f"RIGHT: {r_str}", (two_thirds + 15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.60, r_col, 2)

# Vạch ranh giới nguy hiểm (Danger Line)
line_color = (0, 0, 255) if center_blocked else ((0, 165, 255) if (center_caution or is_quality_degraded) else (0, 255, 255))
cv2.line(frame, (0, danger_y), (w, danger_y), line_color, 2)
cv2.putText(frame, "DANGER LINE (GROUND PERSPECTIVE)", (15, danger_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, line_color, 2)

# Khung cảnh báo chất lượng hình ảnh (nếu có)
if quality_warning:
    banner_w = min(int(w * 0.75), 800)
    bx1 = (w - banner_w) // 2
    bx2 = bx1 + banner_w
    cv2.rectangle(frame, (bx1, 55), (bx2, 92), (10, 10, 10), -1)
    cv2.rectangle(frame, (bx1, 55), (bx2, 92), (0, 165, 255), 2)
    cv2.putText(frame, quality_warning, (bx1 + 15, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

# Khung thông báo vị trí vật cản & trạng thái an toàn
badge_w = min(int(w * 0.90), 920)
stat_x1 = max(10, (w - badge_w) // 2)
stat_x2 = stat_x1 + badge_w
stat_y1 = danger_y + 15
stat_y2 = danger_y + 60

cv2.rectangle(frame, (stat_x1, stat_y1), (stat_x2, stat_y2), (20, 20, 20), -1)
cv2.rectangle(frame, (stat_x1, stat_y1), (stat_x2, stat_y2), status_color, 2)
cv2.putText(frame, status_text, (stat_x1 + 10, stat_y1 + 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 2)

# ==========================================
# 9. FULLSCREEN DISPLAY (ESC TO EXIT)
# ==========================================
WINDOW_NAME = "Visual Assistance System - AI2026 Optimized Mode"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cv2.imshow(WINDOW_NAME, frame)
auto_close_ms = int(os.environ.get("AUTO_CLOSE_MS", "0"))
if auto_close_ms > 0:
    cv2.waitKey(auto_close_ms)
else:
    while True:
        if cv2.waitKey(1) & 0xFF == 27:
            break

cv2.destroyAllWindows()