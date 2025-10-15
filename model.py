from ultralytics import YOLO

# Load YOLOv8n model
model = YOLO("best.pt")

# Run inference on webcam
results = model(
    source=0,      # integer not string
    show=True,     # pop-up window
    conf=0.4,
    save=True
)
