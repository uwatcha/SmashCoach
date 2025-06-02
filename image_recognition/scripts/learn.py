from ultralytics import YOLO

# ベースとするモデル
model = YOLO('image_recognition/models/yolo11n.pt')

# M1 macのGPUを使ってモデルを学習
results = model.train(
    data='datasets/smash-Mario.v1i.yolov11/data.yaml', 
    epochs=3, 
    imgsz=640, 
    device='mps'
)
