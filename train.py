from ultralytics import YOLO

def main():
    # Load a pretrained YOLOv8 nano model (good fit for 6GB VRAM)
    model = YOLO("yolov8n.pt")

    # Fine-tune on your PPE dataset
    model.train(
        data="PPEs-8/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,        # use your GPU
        workers=4,
        patience=10,      # stop early if no improvement for 10 epochs
        project="runs",   # where results get saved
        name="ppe_v1"     # this run's folder name
    )

if __name__ == "__main__":
    main()