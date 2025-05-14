from ultralytics import YOLO 
from paddleocr import PaddleOCR
import cv2
import re
import pandas as pd 
import time
import numpy as np
import os
import datetime
import glob
from shutil import rmtree
import logging
from resource.lib.save_write import *
import resource.lib.logger as logger
import sqlite3

logger.setup_logger('log')
log = logging.getLogger('log')
model = YOLO("C:\\AI_training\\detect\\train17\\weights\\best.pt")

i = 0
img_rate=0.6
img_show_rate=0.4
log.info("Successfully Imported Models")

def cut_box(img0:np.ndarray) -> np.ndarray:
    start = time.time()
    '''從圖案中心裁切'''  
    img=img0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #轉為灰階圖
    ret, img = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY) #二值化：如果大於 88 就等於 255，反之等於 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #取出二值圖的輪廓
    areas = [cv2.contourArea(c) for c in contours]#1
    max_index = areas.index(max(areas))#2
    max_rect =cv2.minAreaRect(contours[max_index])#3
    max_box = cv2.boxPoints(max_rect)#4
    #max_box = np.int0(max_box)#5
    x, y, h, w =cv2.boundingRect(max_box) #取得包覆指定輪廓點的最小正矩形 #b = event.key[1].split("_") #分割符號為 "-"
    if x>=0 and y>=0 : img0 = img0[y:y+w, x: x+h] #裁切所需要的範圍 #imgy=img0.shape[0]      #imgx=img0.shape[1]   #if imgy > imgx  : 
    return img0
        
 
def trans_square(img:np.ndarray) -> np.ndarray:
    '''圖片轉正方形邊緣使用0填充'''
    img_h, img_w, img_c = img.shape
    if img_h != img_w:
        long_side = max(img_w, img_h)
        short_side = min(img_w, img_h)
        loc = abs(img_w - img_h) // 2
        img = img.transpose((1, 0, 2)) if img_w < img_h else img
        background = np.zeros((long_side, long_side, img_c), dtype=np.uint8)  
        background[loc: loc + short_side] = img[...]  
        img = background.transpose((1, 0, 2)) if img_w < img_h else background
    return img


def yolo(img0:np.ndarray) -> int:
    '''切割效期字樣區塊'''
    results1 = model.predict(img0, max_det = 1,conf = 0.3,
                             iou = 0.1 , line_width=1)
    boxs = results1[0].boxes.xyxy
    for result in results1:
        for c in result.boxes.cls:
            cl = str(int(c))

    box=boxs[0] 
    x1 = int(box[0]-8)
    y1 = int(box[1]-5)
    x2 = int(box[2]+20)
    y2 = int(box[3]+6)
    
    #轉灰階、旋轉、分割製造日期(class2-5)
    if cl == '0': #一般圖片
        img_lab=cv2.cvtColor(img0[y1:y2,x1:x2].copy(), cv2.COLOR_RGB2GRAY)
    elif cl == '1': #反向圖片
        img_lab=cv2.cvtColor(img0[y1:y2,x1:x2].copy(), cv2.COLOR_RGB2GRAY)
        img_lab = cv2.rotate(img_lab, cv2.ROTATE_180)
    elif cl == '2': #延伸型
        y2 = y2 - int(abs(y2 - y1)/2)
        img_lab=cv2.cvtColor(img0[y1:y2,x1:x2].copy(), cv2.COLOR_RGB2GRAY)
    elif cl == '3': #延伸+反向
        y2 = y2 - int(abs(y2 - y1)/2)
        img_lab=cv2.cvtColor(img0[y1:y2,x1:x2].copy(), cv2.COLOR_RGB2GRAY)
        img_lab = cv2.rotate(img_lab, cv2.ROTATE_180)
    elif cl == '4': #疊加
        x1 = x1 + int(abs(x2-x1)/2)
        img_lab=cv2.cvtColor(img0[y1:y2,x1:x2].copy(), cv2.COLOR_RGB2GRAY)
    elif cl == '5': #疊加+反向
        x1 = x1 + int(abs(x2-x1)/2)
        img_lab=cv2.cvtColor(img0[y1:y2,x1:x2].copy(), cv2.COLOR_RGB2GRAY)
        img_lab = cv2.rotate(img_lab, cv2.ROTATE_180)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (4, 6))
    img_Kernal = cv2.erode(img_lab, kernel)

    return img_Kernal, x1, x2, y1, y2, cl

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    end=time.time()
    print("Size=" + x1, x2, y1, y2,"Totaltime=" + str(totaltime)+'ms')
    cv2.imshow('Live', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cap.release()
    cv2.destroyAllWindows()
    df = pd.DataFrame(all_data)
    df.to_csv("record.csv", index=False)
    log.info(" Detection record saved to record.csv")