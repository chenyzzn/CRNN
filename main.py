from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np
import os
import glob
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('log')
model = YOLO("C:\\AI_training\\detect\\train17\\weights\\best.pt")

i = 0
img_rate=0.6
img_show_rate=0.4
log.info("Successfully Imported Models")

def detect(img, pixel_per_cm=None):
    results = model.predict(img, max_det=1, conf=0.3, iou=0.1)[0]

    object_data = []
    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        cls = int(cls)
        x1, y1, x2, y2 = map(int, box)
        w_pixel = x2 - x1
        h_pixel = y2 - y1
        w_cm = h_cm = None

        if pixel_per_cm:
            w_cm = round(w_pixel / pixel_per_cm, 2)
            h_cm = round(h_pixel / pixel_per_cm, 2)

        object_data.append({
            'class': cls,
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'width_px': w_pixel, 'height_px': h_pixel,
            'width_cm': w_cm, 'height_cm': h_cm
        })

    return object_data, results.orig_img

# ===== 校正 pixel/cm 的函數（從 class 99）=====
def adjust(objects, known_width_cm=5.0):
    for obj in objects:
        if obj['class'] == 99:  # 校正參考物體的類別編號
            pixel_width = obj['width_px']
            return pixel_width / known_width_cm
    return None

def process_image(img_path, pixel_per_cm=None):
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    objects, _ = detect(img_rgb, pixel_per_cm)

    if not objects:
        return None, img_bgr, pixel_per_cm

    if pixel_per_cm is None:
        pixel_per_cm = adjust(objects)
        if pixel_per_cm:
            log.info(f"Calibrated pixel_per_cm = {pixel_per_cm:.2f}")
        else:
            log.warning("Reference object not found for calibration.")

    # 可視化標記
    for obj in objects:
        label = f"Class {obj['class']}: {obj['width_cm']}cm x {obj['height_cm']}cm" if obj['width_cm'] else f"Class {obj['class']}"
        cv2.rectangle(img_bgr, (obj['x1'], obj['y1']), (obj['x2'], obj['y2']), (0, 255, 0), 2)
        cv2.putText(img_bgr, label, (obj['x1'], obj['y1'] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return objects, img_bgr, pixel_per_cm

if __name__ == '__main__':
    folder = "C:\\AI_training\\test\\*.jpg"
    all_data = []
    pixel_per_cm = None

    for img_path in glob.glob(folder):
        log.info(f"Processing {img_path}")
        objects, annotated_img, pixel_per_cm = process_image(img_path, pixel_per_cm)

        if objects:
            for obj in objects:
                all_data.append({
                    'img_name': os.path.basename(img_path),
                    'class': obj['class'],
                    'width_cm': obj['width_cm'],
                    'height_cm': obj['height_cm'],
                    'x1': obj['x1'], 'y1': obj['y1'], 'x2': obj['x2'], 'y2': obj['y2']
                })

        cv2.imshow("Detected", annotated_img)
        cv2.waitKey(500)

    cv2.destroyAllWindows()

    df = pd.DataFrame(all_data)
    df.to_csv("record.csv", index=False)
    log.info(" Detection record saved to record.csv")
