import cv2
from ultralytics import YOLO
import numpy as np
import supervision as sv

def run_detection(image_path, output_path):

    

    # Load the pre-trained YOLOv8 model from Ultralytics
    model = YOLO('runs\detect\\train\weights\\best.pt')  # You can replace 'yolov8n.pt' with any other model variant

    # Perform detection using the model
    model.predict('example.jpg', save=True, show_boxes = True)

# def process_frame(frame: np.ndarray, _) -> np.ndarray:
#     results = model(frame, imgsz=1280)[0]
    
#     detections = sv.Detections.from_ultralytics(results)

#     box_annotator = sv.BoundingBoxAnnotator(thickness=4)
#     label_annotator = sv.LabelAnnotator()
#     labels = [f"{model.names[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, _, _ in detections]
#     # labels = [f"{model.names[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, _, _ in detections]
#     frame = box_annotator.annotate(scene=frame, detections=detections)
#     frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

#     return frame

def process_frame(frame: np.ndarray, _) -> np.ndarray:
    # Detect objects in the frame
    results = model(frame, size=1280)  # Ensure you adjust size or use default settings
    
    # Convert results from Ultralytics format to Supervision detections
    detections = sv.Detections.from_ultralytics(results.xyxy[0])

    # Initialize annotators for bounding boxes and labels
    box_annotator = sv.BoundingBoxAnnotator(thickness=4)
    label_annotator = sv.LabelAnnotator()

    # Prepare to collect bounding box coordinates
    boxes = []

    # Iterate over detections to extract bounding boxes and labels
    for det in detections:
        x1, y1, x2, y2, conf, class_id = det
        # Format and save the coordinates
        boxes.append((x1, y1, x2, y2, conf, model.names[int(class_id)]))
        
        # Optional: Print each box's coordinates and class
        print(f"Detected {model.names[int(class_id)]} with confidence {conf:.2f} at [{x1}, {y1}, {x2}, {y2}]")

    # Annotate frame with bounding boxes and labels
    labels = [f"{name} {conf:.2f}" for (_, _, _, _, conf, name) in boxes]
    frame = box_annotator.annotate(scene=frame, detections=detections)
    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

    # Optionally, do something with `boxes` like return or store them
    return frame

def process_image(image_path, model):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image could not be read.")
        return

    # Perform detection using the model
    results = model(image)[0]  # Ensure size matches your needs or model's defaults

    # Convert results from Ultralytics format to Supervision detections
    detections = sv.Detections.from_ultralytics(results)

    # Initialize annotators for bounding boxes and labels
    box_annotator = sv.BoundingBoxAnnotator(thickness=4)
    label_annotator = sv.LabelAnnotator()

    # Prepare to collect bounding box coordinates
    boxes = []

    # Iterate over detections to extract bounding boxes and labels
    # for det in detections:
    #     x1, y1, x2, y2, conf, class_id = det
    #     boxes.append((x1, y1, x2, y2, conf, model.names[int(class_id)]))

    #     # Optional: Print each box's coordinates and class
    #     print(f"Detected {model.names[int(class_id)]} with confidence {conf:.2f} at [{x1}, {y1}, {x2}, {y2}]")

    # Annotate image with bounding boxes and labels
    # labels = [f"{name} {conf:.2f}" for (_, _, _, _, conf, name) in boxes]
    annotated_image = box_annotator.annotate(scene=image, detections=detections)
    # annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    return annotated_image

if __name__ == "__main__":
    # Path to your input image file and output file
    model = YOLO('runs\detect\\train\weights\\best.pt')
    VIDEO_PATH = 'video1.mp4'
    video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)
    image_path = 'frame1.png'  # Replace with your image path
    output_path = 'output.jpg'  # Replace with desired output path
    # run_detection(image_path, output_path)
    # sv.process_video(source_path=VIDEO_PATH, target_path=f"result.mp4", callback=process_frame)
    processed_image = process_image(image_path, model)
    cv2.imshow('Processed Image', processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# python yolov8_sahi.py --source "../../video.mp4" --save-img --weights "../../best.pt"


# Assuming detect_objects is your function to run the YOLO model
# This function should return a list of detections, where each detection is a dictionary
# Example of a detection: {'class_id': 1, 'confidence': 0.86, 'box': (x, y, w, h)}

image = cv2.imread('path_to_your_image.jpg')
detections = run_detection(image)

for detection in detections:
    class_id = detection['class_id']
    confidence = detection['confidence']
    (x, y, w, h) = detection['box']
    
    # Now you can use x, y, w, h to work with the bounding box
    print(f"Class ID: {class_id}, Confidence: {confidence}, BBox: {x}, {y}, {w}, {h}")

    # If you want to draw the bounding box on the image:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow("Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()