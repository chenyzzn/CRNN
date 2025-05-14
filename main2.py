from ultralytics import YOLO 
import cv2
import pandas as pd
import numpy as np
import os
import glob
import logging
import time 

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('log')
model = YOLO("C:\\AI_training\\detect\\train17\\weights\\best.pt")

i = 0
img_rate = 0.6
img_show_rate = 0.4

def detect(img, pixel_per_cm=None):
    results1 = model.predict(img, max_det=1, conf=0.3, iou=0.1)
    boxs = results1[0].boxes.xyxy

    object_data = []
    for box in boxs:
        x1 = int(box[0] - 8)
        y1 = int(box[1] - 5)
        x2 = int(box[2] + 20)
        y2 = int(box[3] + 6)
        w_pixel = x2 - x1
        h_pixel = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        object_data.append({
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'width_px': w_pixel, 'height_px': h_pixel,
            'center_x': center_x, 'center_y': center_y
        })

    return object_data, results1[0].orig_img

def process_image(img_path, pixel_per_cm=None):
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    objects, _ = detect(img_rgb, pixel_per_cm)

    for obj in objects:
        label = f"Width: {obj['width_px']} px, Height: {obj['height_px']} px"
        cv2.rectangle(img_bgr, (obj['x1'], obj['y1']), (obj['x2'], obj['y2']), (0, 255, 0), 2)
        cv2.putText(img_bgr, label, (obj['x1'], obj['y1'] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return objects, img_bgr, pixel_per_cm

if __name__ == '__main__':
    all_data = []
    pixel_per_cm = None
    cap = cv2.VideoCapture(0)
    start = time.time()

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, frame = cap.read()
        time.sleep(0.1)
        if not ret:
            print("Cannot receive frame")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray_frame, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 計算中心點
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        objects, annotated_img = detect(frame, pixel_per_cm)

        if objects:
            for obj in objects:
                all_data.append({
                    'img_name': 'webcam_frame',
                    'x1': obj['x1'], 'y1': obj['y1'],
                    'x2': obj['x2'], 'y2': obj['y2'],
                    'cx': obj['center_x'], 'cy': obj['center_y']
                })

                cv2.rectangle(frame, (obj['x1'], obj['y1']), (obj['x2'], obj['y2']), (0, 255, 0), 2)
                cv2.putText(frame, f"({obj['x1']},{obj['y1']})", (obj['x1'], obj['y1'] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(frame, f"({obj['x2']},{obj['y2']})", (obj['x2'], obj['y2'] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(frame, f"Center: ({int(obj['center_x'])}, {int(obj['center_y'])})", (int(obj['center_x']), int(obj['center_y'])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        end = time.time()
        print("Total Time:", end - start)
        cv2.imshow('Live', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    df = pd.DataFrame(all_data)
    df.to_csv("record.csv", index=False)
    log.info("Detection record saved to record.csv")
