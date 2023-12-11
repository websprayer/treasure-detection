from ultralytics import YOLO
if __name__ == '__main__':
# Load a model
    model = YOLO('best.pt')  # load a pretrained model (recommended for training)
# Train the model with 2 GPUs
    model.train(data='./dataset/data.yaml', epochs=150, imgsz=640,batch=20)