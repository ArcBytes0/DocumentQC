from ultralytics import YOLO

# Load a pre-trained YOLO model
model = YOLO('yolov8n.pt')

# Train the model on a custom dataset
model.train(
    data='data/data.yaml',  # Path to dataset configuration file
    epochs=50,                  # Number of training epochs
    imgsz=640,                  # Image size
    batch=16,                   # Batch size
    name='doc_qc_model',      # Name for the training run
)

# Export best weights
model.export(format='pt')  # Export the best model weights to ONNX format
print("Model training and export completed.")