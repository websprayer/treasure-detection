import cv2
import torch
from ultralytics import YOLO
import sys
import numpy as np  # 导入NumPy库
from PIL import ImageGrab
import threading
import time
import os
import shutil
# 加载YOLOv5模型
model = YOLO('best.pt')  # load a pretrained model (recommended for training)
# 源文件夹路径
source_folder = 'liang'

# 目标文件夹路径
destination_folder = 'ji'
# Video parameters
output_path = 'output_video.avi'  # Output video path (AVI format)
fps = 30  # Video frame rate
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID codec for AVI format

# Get screen resolution
screen_width, screen_height = 1920, 1080  # Adjust as needed

# Create VideoWriter object
out = cv2.VideoWriter(output_path, fourcc, fps, (screen_width, screen_height))

# Flag to indicate program exit
exit_program = False

# Function to handle keyboard input
def check_keyboard_input():
    global exit_program
    while True:
        if input('Press "q" to quit: ').strip().lower() == 'q':
            exit_program = True
            break

# Create and start a thread to handle keyboard input
keyboard_thread = threading.Thread(target=check_keyboard_input)
keyboard_thread.daemon = True  # Set the thread as a daemon
keyboard_thread.start()
i=0
while not exit_program:
    # Capture screen image
    im = ImageGrab.grab(bbox=(0, 0, screen_width, screen_height))
    im_np = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    # Use YOLOv8 for object detection
    im_path_virtual = "current_screen_image"
    results = model(im_np,conf=0.4,save=True,project="liang",exist_ok=True)  # Perform detection with YOLOv8
    results_path =  im_path_virtual
    pred = results[0] if len(results) > 0 else None
    # boxes = results.boxes if hasattr(results, 'boxes') else None
    # if boxes is not None and len(boxes.tensor) > 0:
    #    for det in boxes.tensor[0]:
    #      xmin, ymin, xmax, ymax, conf, label = det['xmin'], det['ymin'], det['xmax'], det['ymax'], det['conf'], det['label']
    #      cv2.rectangle(im_np, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0 , 255), 2)
    #      cv2.putText(im_np, f'{model.names[int(label)]} {conf:.2f}', (int(xmin), int(ymin) - 10),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # out.write(im_np)
    # # Display detection results
# 遍历主文件夹下的所有子文件夹
    for subdir in os.listdir('liang'):
     subdir_path = os.path.join('liang', subdir)
     if os.path.isdir(subdir_path):
        # 遍历子文件夹中的所有文件
        for file in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, file)
            # 判断文件是否是图片（可以根据实际情况修改判断条件）
            if file_path.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                # 读取图片
                img = cv2.imread(file_path)
                if img is not None:
                    # 调整图片大小
                    img = cv2.resize(img, (screen_width, screen_height))
                    # 写入视频
                    out.write(img)
                    new_filename = f"img_{i}.jpg"
                    i=i+1
                    destination_file = os.path.join(destination_folder, new_filename)
                    # 移动并重命名文件
                    shutil.move(file_path, destination_file)
    # 移动文件到目标文件夹
    
    #cv2.imshow('YOLOv8 Object Detection', im_np)
    #cv2.waitKey(100)  # Delay for 100 milliseconds
    print("检测结果:", results)
    result= None
# Release resources
cv2.destroyAllWindows()
out.release()  # Release the video writer
print('Video saved successfully.')