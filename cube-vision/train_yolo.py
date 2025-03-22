from ultralytics import YOLO

def train_yolo():
    # Load existing trained model (Rubik's Cube detection model)
    model = YOLO("best.pt")  # Ensure best.pt is the correct model

    # Train the model on the sticker dataset
    model.train(
        data="C:/Users/bs840/Desktop/Study/CS291I/final_project/cube-vision/yolo_sticker/data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        lr0=0.01,
        device="cuda",
        mosaic=1.0,
    )

def train_scratch():
    # Load existing trained model (Rubik's Cube detection model)
    model = YOLO("yolov8n.pt")

    # Train the model on the sticker dataset
    model.train(
        data="C:/Users/bs840/Desktop/Study/CS291I/final_project/cube-vision/yolo_sticker/data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        lr0=0.01,
        workers=4,
        device="cuda",
        mosaic=1.0
    )


def test_yolo():
    # model = YOLO("best.pt")
    model = YOLO("runs/detect/train18/weights/best.pt")
    model.val()

    model.predict(source="C:/Users/bs840/Desktop/Study/CS291I/final_project/cube-vision/yolo_sticker/test/images", save=True, conf=0.4)


if __name__ == "__main__":
    # train_yolo()
    train_scratch()

    # test_yolo()
