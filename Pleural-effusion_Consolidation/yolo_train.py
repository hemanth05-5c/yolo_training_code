from ultralytics import YOLO

model = YOLO("yolov8m-seg.pt")

model.train(data="/home/ai-user/RELA PROJECT/Pleural-effusion_Consolidation/yolo_dataset/dataset.yaml", epochs=800, batch=32)