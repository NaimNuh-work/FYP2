from ultralytics import YOLO

if __name__ == '__main__':
    print("[INFO] Loading base YOLOv8 model...")
    # We start with the 'nano' weights. It already knows basic shapes,
    # we are just teaching it to specialize in cones.
    model = YOLO("yolov8n.pt")

    print("[INFO] Starting Training Phase...")

    # --- TRAINING PARAMETERS ---
    results = model.train(
        data="dataset/data.yaml",  # UPDATE THIS: Put the exact path to your data.yaml file
        epochs=50,  # How many times it loops through the dataset. 50 is a good start.
        imgsz=640,  # Standard image size for training
        batch=16,  # How many images it processes at once
        device=0,  # Forces the system to use your NVIDIA RTX 3050 GPU for maximum speed
        plots=True,  # Generates loss graphs and validation charts for your final thesis report
        name="cone_detector"  # The name of the folder where results will be saved
    )

    print("\n[SUCCESS] Training Complete!")
    print("Your custom AI model is saved at: runs/detect/cone_detector/weights/best.pt")