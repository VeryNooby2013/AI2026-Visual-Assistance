import time
import os
import glob
import cv2
import numpy as np
from ultralytics import YOLOWorld

# Ground truth tập trung vào phát hiện nguy hiểm làn giữa & chất lượng
# ĐẶC BIỆT: Tủ điện, hàng rào, đường trống
GROUND_TRUTH = {
    "istockphoto-1203937499-612x612.jpg": {
        "quality": "NORMAL", "left": False, "center": False, "right": False,
        "is_empty_path": True, "desc": "Đường trống hoàn toàn"
    },
    "adversarial_test.jpg": {
        "quality": "BLURRY", "left": False, "center": False, "right": False,
        "is_empty_path": False, "desc": "Ảnh mờ test"
    },
    "test_dark.jpg": {
        "quality": "DARK", "left": False, "center": True, "right": False,
        "is_empty_path": False, "desc": "Ảnh tối có xe cản giữa"
    },
    "test0.jpg": {
        "quality": "NORMAL", "left": False, "center": True, "right": False,
        "is_empty_path": False, "desc": "Xe đạp/người cản giữa"
    },
    "10.jpg": {
        "quality": "NORMAL", "left": True, "center": False, "right": False,
        "is_empty_path": False, "desc": "Thùng phi lề trái, giữa thông"
    },
    "aaa.jpg": {
        "quality": "NORMAL", "left": True, "center": True, "right": False,
        "is_barrier": True, "desc": "Hàng rào chắn lấn giữa >= 10%"
    },
    "images.jpg": {
        "quality": "NORMAL", "left": False, "center": False, "right": False,
        "is_empty_path": True, "desc": "Đường thoáng, cản xa"
    },
    "test.jpg": {
        "quality": "NORMAL", "left": False, "center": False, "right": False,
        "is_empty_path": True, "desc": "Đường trước thông thoáng"
    },
    "999.jpg": {
        "quality": "NORMAL", "left": False, "center": True, "right": True,
        "is_empty_path": False, "desc": "Ô tô & thùng phi giữa-phải"
    },
    "1test.jpg": {
        "quality": "NORMAL", "left": True, "center": True, "right": True,
        "is_empty_path": False, "desc": "Thùng phi chắn lối"
    },
    "images (1).jpg": {
        "quality": "NORMAL", "left": True, "center": True, "right": False,
        "is_empty_path": False, "desc": "Xe máy/người cản trái-giữa"
    },
    "images (2).jpg": {
        "quality": "NORMAL", "left": True, "center": True, "right": True,
        "is_empty_path": False, "desc": "Xe chắn kín"
    },
    "images (3).jpg": {
        "quality": "NORMAL", "left": True, "center": True, "right": True,
        "is_empty_path": False, "desc": "Ô tô chắn giữa"
    },
    "duong_pho.jpg": {
        "quality": "NORMAL", "left": True, "center": True, "right": True,
        "is_empty_path": False, "desc": "Đường đông người xe"
    },
    "pho-tay-ha-noi.jpg": {
        "quality": "NORMAL", "left": True, "center": True, "right": True,
        "is_empty_path": False, "desc": "Phố đi bộ đông người"
    },
    "9b4c38b5-80a7-45a1-8beb-e72298914694.jpg": {
        "quality": "NORMAL", "left": True, "center": True, "right": True,
        "is_empty_path": False, "desc": "Đường đông người xe"
    },
    "user_test_cabinet_close.jpg": {
        "quality": "NORMAL", "left": False, "center": True, "right": False,
        "is_cabinet": True, "desc": "Tủ điện lớn choán kín giữa cận kề"
    },
    "user_test_cabinet_far.jpg": {
        "quality": "NORMAL", "left": True, "center": True, "right": True,
        "is_cabinet": True, "desc": "Tủ điện lớn làn giữa phía trước"
    }
}

CUSTOM_CLASSES = [
    "cabinet", "electric box", "obstacle", "barrier", "fence",
    "box", "barrel", "sign", "person", "vehicle", "wall", "pillar"
]
CONF_CONFIRMED = 0.18
CONF_UNCERTAIN = 0.08

def evaluate_imgsz(imgsz, images, warm_up=True):
    # Đo thời gian khởi tạo riêng biệt
    t0_init = time.perf_counter()
    model = YOLOWorld('C:/Users/MSI/PycharmProjects/HTKhiemThi/yolov8s-world.pt')
    model.set_classes(CUSTOM_CLASSES)
    # Ép chạy trên CPU
    import torch
    device = 'cpu'
    t_init = (time.perf_counter() - t0_init) * 1000 # ms

    # Warm-up 1 lần để CPU load cache/jit
    if warm_up and len(images) > 0:
        dummy = cv2.imread(images[0])
        model.predict(dummy, conf=CONF_UNCERTAIN, iou=0.40, imgsz=imgsz, device=device, verbose=False)

    inference_times = []
    
    tp, fp, fn, tn = 0, 0, 0, 0
    cabinet_detected = 0
    cabinet_total = 2  # close & far
    barrier_detected = 0
    barrier_total = 1  # aaa.jpg
    empty_false_alarms = 0
    empty_total = 3    # istockphoto, images.jpg, test.jpg

    details = []

    for p in images:
        name = os.path.basename(p)
        frame = cv2.imread(p)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        one_third = w // 3
        two_thirds = 2 * (w // 3)
        danger_y = int(h * 0.46)

        # Đo thời gian suy luận riêng biệt
        t0_inf = time.perf_counter()
        results = model.predict(frame, conf=CONF_UNCERTAIN, iou=0.40, imgsz=imgsz, device=device, verbose=False)[0]
        t_inf = (time.perf_counter() - t0_inf) * 1000 # ms
        inference_times.append(t_inf)

        center_blocked = False
        center_caution = False
        left_blocked = False
        right_blocked = False
        left_caution = False
        right_caution = False

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confs, clss):
                x1, y1, x2, y2 = box
                box_w = max(x2 - x1, 1)
                box_h = max(y2 - y1, 1)
                area = box_w * box_h

                if y2 < int(h * 0.20) or area < (w * h * 0.002):
                    continue

                is_confirmed = conf >= CONF_CONFIRMED
                cx = (x1 + x2) // 2

                # Logic 10% lấn làn giữa giữ nguyên
                overlap_left = max(x1, one_third)
                overlap_right = min(x2, two_thirds)
                mid_overlap = max(0, overlap_right - overlap_left)
                overlap_ratio = mid_overlap / box_w

                in_center = (overlap_ratio >= 0.10) or (one_third <= cx <= two_thirds)
                is_danger = (y2 >= danger_y) or (in_center and box_h > int(h * 0.30))

                left_overlap = max(0, min(x2, one_third) - max(x1, 0))
                right_overlap = max(0, min(x2, w) - max(x1, two_thirds))
                in_left = (left_overlap / box_w >= 0.15) or (cx < one_third)
                in_right = (right_overlap / box_w >= 0.15) or (cx > two_thirds)

                if is_danger:
                    if in_center:
                        if is_confirmed: center_blocked = True
                        else: center_caution = True
                    if in_left:
                        if is_confirmed: left_blocked = True
                        else: left_caution = True
                    if in_right:
                        if is_confirmed: right_blocked = True
                        else: right_caution = True

        # Tầng dự phòng tủ điện / vật thể to giữ nguyên
        if not center_blocked:
            mid_roi = frame[int(h * 0.10):int(h * 0.85), one_third:two_thirds]
            if mid_roi.size > 0:
                gray_roi = cv2.cvtColor(mid_roi, cv2.COLOR_BGR2GRAY)
                _, white_mask = cv2.threshold(gray_roi, 170, 255, cv2.THRESH_BINARY)
                white_pixels = cv2.countNonZero(white_mask)
                roi_area = mid_roi.shape[0] * mid_roi.shape[1]
                white_ratio = white_pixels / roi_area
                if white_ratio > 0.28:
                    center_blocked = True

        gt = GROUND_TRUTH.get(name, {"quality": "NORMAL", "left": False, "center": False, "right": False})
        lanes = [
            ("L", left_blocked or left_caution, gt["left"]),
            ("C", center_blocked or center_caution, gt["center"]),
            ("R", right_blocked or right_caution, gt["right"])
        ]
        for l_name, pred, actual in lanes:
            if pred and actual: tp += 1
            elif pred and not actual: fp += 1
            elif not pred and actual: fn += 1
            else: tn += 1

        # Đánh giá riêng cho Tủ điện, Hàng rào, Đường trống
        c_has = center_blocked or center_caution
        if gt.get("is_cabinet", False) and c_has:
            cabinet_detected += 1
        if gt.get("is_barrier", False) and c_has:
            barrier_detected += 1
        if gt.get("is_empty_path", False) and c_has:
            empty_false_alarms += 1

        details.append((name, t_inf, c_has, gt["center"]))

    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    avg_inf = np.mean(inference_times)
    fps = 1000.0 / avg_inf if avg_inf > 0 else 0

    return {
        "imgsz": imgsz,
        "init_ms": t_init,
        "avg_inf_ms": avg_inf,
        "min_inf_ms": np.min(inference_times),
        "max_inf_ms": np.max(inference_times),
        "fps": fps,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "cabinet": f"{cabinet_detected}/{cabinet_total}",
        "barrier": f"{barrier_detected}/{barrier_total}",
        "empty_fa": f"{empty_false_alarms}/{empty_total}"
    }

if __name__ == '__main__':
    images = sorted(glob.glob('C:/Users/MSI/PycharmProjects/HTKhiemThi/*.jpg'))
    # Thêm user_test_barrier.png nếu có
    png_path = 'C:/Users/MSI/PycharmProjects/HTKhiemThi/user_test_barrier.png'
    if os.path.exists(png_path) and png_path not in images:
        images.append(png_path)
        
    print(f"BẮT ĐẦU BENCHMARK TRÊN CPU VỚI {len(images)} ẢNH THỰC TẾ...\n")
    sizes = [1280, 960, 640]
    results = []
    for s in sizes:
        print(f"-> Đang đo imgsz = {s}...")
        res = evaluate_imgsz(s, images)
        results.append(res)
        print(f"   Hoàn thành imgsz={s}: Khởi tạo={res['init_ms']:.1f}ms, Suy luận tb={res['avg_inf_ms']:.1f}ms ({res['fps']:.2f} FPS)")

    print("\n" + "=" * 80)
    print("                     BẢNG TỔNG HỢP BENCHMARK TRÊN CPU")
    print("=" * 80)
    header = f"{'imgsz':<6} | {'Init (ms)':<10} | {'Inf TB (ms)':<11} | {'FPS':<6} | {'Precision':<10} | {'Recall':<8} | {'F1':<6} | {'Tủ điện':<8} | {'H.rào':<6} | {'Báo nhầm trống':<15}"
    print(header)
    print("-" * 80)
    for r in results:
        print(f"{r['imgsz']:<6} | {r['init_ms']:<10.1f} | {r['avg_inf_ms']:<11.1f} | {r['fps']:<6.2f} | {r['precision']*100:<9.1f}% | {r['recall']*100:<7.1f}% | {r['f1']*100:<5.1f}% | {r['cabinet']:<8} | {r['barrier']:<6} | {r['empty_fa']:<15}")
    print("=" * 80)
