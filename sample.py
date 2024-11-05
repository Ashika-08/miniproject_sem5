from ultralytics import YOLO

# Load YOLOv8 model (pre-trained)
model = YOLO('yolov8n.pt')  # use 'yolov8s.pt' for a larger model, if needed

# Train the model
model.train(
    data='dataset/data.yaml',    # path to the YAML file you created
    epochs=50,                # number of epochs to train
    imgsz=640,                # image size for training
    batch=16,                 # batch size
    name='yolov8_acne_skin',  # name for your training session
    workers=4                 # number of workers for data loading
)
