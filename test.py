from ultralytics import YOLO
path="D:/codes/python/version/lyw1/testImage/"
model = YOLO('best.pt')
result=model.predict(path,save_txt=True,save = True,conf=0.4)