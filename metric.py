import time
import os
import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon
from PIL import Image, ImageDraw
from IPython.display import display
from tabulate import tabulate
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment


results_summary = []

COLOR_MAP = {
    "white":   (255, 255, 255),
    "yellow":  (0, 255, 255),
    "blue":    (255, 0, 0),
    "green":   (0, 128, 0),
    "red":     (0, 0, 255),
    "orange":  (0, 165, 255)
}
class_names = {
    0: "blue",
    1: "center",
    2: "green",
    3: "orange",
    4: "red",
    5: "white",
    6: "yellow"
}

def test_yolo(image_path, model):
    start_time = time.time()
    results = model.predict(source=image_path, conf=0.4, save=False)
    end_time = time.time()

    # Extract bounding boxes and detected colors
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)  # Bounding box format
            class_id = int(box.cls[0].cpu().numpy())  # Class ID
            detections.append((class_id, [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]))  # Convert bbox to polygon format
    # for result in results:
    #     if result.masks is None:
    #         print("skipped")
    #         continue  # Skip if no masks are detected
    #     print(result)
    #     masks = result.masks.xy  # Extracts polygon masks from YOLO detection
    #     class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Extract class IDs

    #     for mask, class_id in zip(masks, class_ids):
    #         polygon = [(int(pt[0]), int(pt[1])) for pt in mask]  # Convert to integer coordinates
    #         detections.append((class_id, polygon))

    inference_time = end_time - start_time
    return detections, inference_time

rc_to_yolo_mapping = {
    0: 0,  # Blue
    2: 2,  # Green
    3: 3,  # Orange
    4: 4,  # Red
    5: 5,  # White
    6: 6   # Yellow
}
def rotate_90(m):
    """
    Rotate the 3x3 matrix m by 90 degrees clockwise.
    m is a list of lists (3 rows, each row 3 items).
    Returns a new 3x3 list of lists.
    """
    # Pythonic approach (zip + reversed).
    # Be mindful it yields tuples, so convert to list.
    return [list(row) for row in zip(*m[::-1])]
def rotate_180(m):
    return rotate_90(rotate_90(m))
def rotate_270(m):
    return rotate_90(rotate_180(m))
def flip_horizontal(m):
    """
    Flip left<->right.
    Each row reversed.
    """
    return [row[::-1] for row in m]
def flip_vertical(m):
    """
    Flip top<->bottom.
    Just reverse the rows order.
    """
    return m[::-1]
def flip_diagonal(m):
    """
    Flip along the main diagonal (top-left to bottom-right).
    After this, m[row][col] => new[col][row].
    """
    # zip(*m) returns tuples, so convert each row to list.
    return [list(x) for x in zip(*m)]
def flip_antidiagonal(m):
    """
    Flip along the anti-diagonal (top-right to bottom-left).
    One way is rotate_180, then flip_diagonal, etc.
    """
    # For example:
    return flip_diagonal(rotate_180(m))

def test_rc(image_path, model):
    start_time = time.time()
    results = model(image_path)
    cls = results[0].boxes.cls
    end_time = time.time()

    masks = results[0].masks
    if masks is None or len(masks.xy) == 0:
        print(f"No Masks detected by Camsolver")
        return [], end_time - start_time
    
    polygon_list = []
    for mask in masks:
        polygon_list.append(mask.xy[0])
    filtered_indices = [i for i, c in enumerate(cls) if c != 1]
    cube_polygons = [polygon_list[i] for i in filtered_indices]
    new_cls = [int(cls[i]) for i in filtered_indices]
    detections = [(class_id, [(int(pt[0]), int(pt[1])) for pt in polygon]) for class_id, polygon in zip(new_cls, cube_polygons)]
    # mapped_detections = [(rc_to_yolo_mapping.get(class_id, -1), polygon) for class_id, polygon in zip(new_cls, cube_polygons)]
    # mapped_detections = [det for det in mapped_detections if det[0] != -1]
    inference_time = end_time - start_time
    return detections, inference_time

def load_ground_truth(image_path, label_folder="cube-vision/yolo_sticker/test/labels", image_size=(640, 640)):
    """
    Loads YOLO ground truth labels and converts them into (x1, y1, x2, y2) format.
    
    Args:
        image_path (str): Path to the image file.
        label_folder (str): Folder where YOLO label files are stored.
        image_size (tuple): (width, height) of the image.

    Returns:
        List of tuples: [(class_id, x1, y1, x2, y2), ...]
    """
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    label_path = os.path.join(label_folder, f"{image_name}.txt")

    if not os.path.exists(label_path):
        print(f"Warning: No ground truth label found for {image_path}")
        return []

    img_w, img_h = image_size  # Image dimensions
    ground_truth = []

    with open(label_path, "r") as file:
        for line in file.readlines():
            parts = list(map(float, line.strip().split()))
            class_id = int(parts[0])

            polygon = []
            coords = parts[1:]
            for i in range(0, len(coords), 2):
                x = int(coords[i] * img_w)  # Convert normalized x to pixels
                y = int(coords[i + 1] * img_h)  # Convert normalized y to pixels
                polygon.append((x, y))
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            bounding_box = (x_min, y_min, x_max, y_max)
            ground_truth.append((class_id, polygon, bounding_box))

    return ground_truth

def calculate_iou_polygon(polygon1, polygon2):
    """
    Calculate Intersection over Union (IoU) between two polygons.

    Args:
        polygon1: List of (x, y) coordinates representing the first polygon.
        polygon2: List of (x, y) coordinates representing the second polygon.

    Returns:
        IoU value (0 to 1).
    """
    poly1 = Polygon(polygon1)
    poly2 = Polygon(polygon2)

    if not poly1.is_valid or not poly2.is_valid:
        return 0  # Invalid polygons do not overlap

    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area

    if union_area == 0:
        return 0  # Avoid division by zero
    iou = intersection_area / union_area
    return intersection_area / union_area

def calculate_iou_bbox(b1, b2):
    """
    b1, b2 are (x1,y1,x2,y2) in pixel coords.
    Returns IoU in [0,1].
    """
    x1, y1, x2, y2 = b1
    x3, y3, x4, y4 = b2

    # intersection
    ix1 = max(x1, x3)
    iy1 = max(y1, y3)
    ix2 = min(x2, x4)
    iy2 = min(y2, y4)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter_area = iw * ih

    # areas
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0

def check_overlap(normals, centers):
    iou_threshold = 0.8
    normal = []
    for n_det in normals:
        is_duplicate = False
        n_box = n_det['bbox']
        for c_det in centers:
            c_box = c_det['bbox']
            iou = calculate_iou_bbox(n_box, c_box)
            if iou > iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            normal.append(n_det)
    return normal

def evaluate_model(model_name, detections, ground_truth, rc, image_name, inference_time):
    total_correct = 0
    total_detections = len(detections)
    total_ground_truth = len(ground_truth)

    for det in detections:
        class_id, det_shape = det
        matching_gt = [(gt_class, gt_polygon, gt_box) for gt_class, gt_polygon, gt_box in ground_truth if gt_class == class_id]
        best_iou = 0.0
        for (gt_class, gt_polygon, gt_box) in matching_gt:
            if rc:
                iou = calculate_iou_polygon(det_shape, gt_polygon)
            else:
                xs = [pt[0] for pt in det_shape]
                ys = [pt[1] for pt in det_shape]
                x_min = min(xs)
                x_max = max(xs)
                y_min = min(ys)
                y_max = max(ys)
                
                iou = calculate_iou_bbox((x_min, y_min, x_max, y_max), gt_box)
            if iou > best_iou:
                best_iou = iou
        
        # max([calculate_iou_polygon(det_polygon, gt_polygon)
        #                  for _, gt_polygon in matching_gt], default=0)
        if best_iou > 0.5:  # IoU > 50% is considered a correct detection
            total_correct += 1

    precision = total_correct / total_detections if total_detections > 0 else 0
    recall = total_correct / total_ground_truth if total_ground_truth > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"{model_name} Performance:")
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}")
    results_summary.append([image_name, model_name, f"{precision:.2f}", f"{recall:.2f}", f"{f1_score:.2f}", f"{inference_time:.4f}s"])

def visualize_detections(image_path, ground_truth, yolo_detections, rc_detections, save_folder="test_dataset/visualization"):
    """
    Draws ground truth, YOLO, and Rubiks_Camsolver detections on the image.
    
    Args:
        image_path (str): Path to the image file.
        ground_truth (list): List of ground truth polygons [(class_id, polygon)].
        yolo_detections (list): List of YOLO detected polygons [(class_id, polygon)].
        rc_detections (list): List of Rubiks_Camsolver detected polygons [(class_id, polygon)].
    """
    img_left = Image.open(image_path).convert("RGB")
    img_right = img_left.copy()

    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)

    # Draw Ground Truth Polygons (Blue)
    for gt_class, polygon, box in ground_truth:
        draw_left.rectangle(box, outline=(0, 0, 255), width=3)
        x1, y1, x2, y2 = box
        draw_left.text((x1, y1 - 10), f"GT {gt_class}", fill=(0, 0, 255))

        draw_right.polygon(polygon, outline=(0, 0, 255), width=3)
        draw_right.text(polygon[0], f"GT {gt_class}", fill=(0, 0, 255))

    # Draw YOLOv8 Detections (Green)
    for det_class, polygon in yolo_detections:
        if det_class == 1:
            print("center found")
            draw_left.polygon(polygon, outline=(255, 0, 0), width=4)
            draw_left.text(polygon[0], "CENTER", fill=(255, 0, 0))
        else:
            draw_left.polygon(polygon, outline=(0, 255, 0), width=3)
            draw_left.text(polygon[0], f"YOLO {det_class}", fill=(0, 255, 0))

    # Draw Rubiks_Camsolver Detections (Red)

    for det_class, polygon in rc_detections:
        draw_right.polygon(polygon, outline=(255, 0, 0), width=3)
        draw_right.text(polygon[0], f"RC {det_class}", fill=(255, 0, 0))

    # Show and save the image
    width, height = img_left.size
    combined_img = Image.new("RGB", (width * 2, height))
    combined_img.paste(img_left, (0, 0))
    combined_img.paste(img_right, (width, 0))
    # combined_img.show()
    filename = os.path.basename(image_path).replace(".jpg", "_comparison.jpg").replace(".png", "_comparison.png")

    save_path = os.path.join(save_folder, filename)
    combined_img.save(save_path)
    # print(f"Saved visualization: {save_path}")

def detect_one_face(frame, model):
    results = model.predict(source=frame, conf=0.5, save=False, verbose=False)
    face_3x3 = []

    if len(results) == 0:
        return False, face_3x3
    
    boxes = results[0].boxes
    if len(boxes) != 9:
        # If we don't get exactly 9 bounding boxes, skip
        return False, face_3x3

    # 2) Extract bounding box coords & class name
    stickers = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        class_id = int(box.cls[0])
        # If the model has named classes:
        if hasattr(results[0], 'names'):
            color_name = results[0].names[class_id]
        else:
            color_name = str(class_id)

        # Convert to integer coords
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

        # We'll store the bounding box center or top-left for sorting
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Keep track of relevant info
        stickers.append({
            'cx': cx,
            'cy': cy,
            'color': color_name,
            'bbox': (x1, y1, x2, y2)
        })

    # 3) Sort the 9 stickers into a 3x3 arrangement
    #    A simple approach: sort primarily by y ascending, secondarily by x ascending
    #    Then slice into rows of 3
    stickers_sorted = sorted(stickers, key=lambda s: (s['cy'], s['cx']))

    # Now group them in rows of 3
    row1 = stickers_sorted[0:3]
    row2 = stickers_sorted[3:6]
    row3 = stickers_sorted[6:9]

    # Within each row, sort by x ascending (if you want left->right)
    row1_sorted = sorted(row1, key=lambda s: s['cx'])
    row2_sorted = sorted(row2, key=lambda s: s['cx'])
    row3_sorted = sorted(row3, key=lambda s: s['cx'])

    # 4) Build the final 3x3 color layout
    face_3x3 = [
        [st['color'] for st in row1_sorted],
        [st['color'] for st in row2_sorted],
        [st['color'] for st in row3_sorted]
    ]

    return True, face_3x3

def show_detection():
    model = YOLO("cube-vision/runs/detect/train19/weights/best.pt")
    # rc_model = YOLO("RubikCubeColorExtractor-main/cube_detector.pt")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
                print("Error: could not read frame")
                break
        results = model(frame, conf = 0.45, verbose=False)

        class_counts = 0
        detections = []  # each item => { 'class_id', 'cx','cy','bbox', 'conf' }
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                class_id = int(box.cls[0])  # Class ID
                conf = float(box.conf[0])  # Confidence score
                class_name = model.names[class_id]  # Get class name from model
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                detections.append({
                    'class_id': class_id,
                    'cx': cx,
                    'cy': cy,
                    'bbox': (x1, y1, x2, y2),
                    'conf': conf,
                    'face_id': None
                })
        centers = [d for d in detections if d['class_id'] == 1]
        normals = [d for d in detections if d['class_id'] != 1]
        normals = check_overlap(normals, centers)
        grouped_faces = []
        if len(centers) == 1:
            center_det = centers[0]
            # find 8 nearest from normals
            distances = []
            for s in normals:
                dx = s['cx'] - center_det['cx']
                dy = s['cy'] - center_det['cy']
                dist_sq = dx*dx + dy*dy
                distances.append((dist_sq, s))
            distances.sort(key=lambda x: x[0])
            # pick the first 8
            nearest_8 = [info[1] for info in distances[:8]] if len(distances)>=8 else [info[1] for info in distances]
            faceA = [center_det] + nearest_8
            if filter_center(center_det, nearest_8):
                grouped_faces.append(faceA)
        elif len(centers) == 2:
            centerA = centers[0]
            dA = []
            for s in normals:
                dx = s['cx'] - centerA['cx']
                dy = s['cy'] - centerA['cy']
                dist_sq = dx*dx + dy*dy
                dA.append((dist_sq, s))
            dA.sort(key=lambda x: x[0])
            nearest_8A = [info[1] for info in dA[:8]] if len(dA)>=8 else [info[1] for info in dA]
            if filter_center(centerA, nearest_8A):
                faceA = [centerA] + nearest_8A
                grouped_faces.append(faceA)
            # Face B
            centerB = centers[1]
            dB = []
            for s in normals:
                dx = s['cx'] - centerB['cx']
                dy = s['cy'] - centerB['cy']
                dist_sq = dx*dx + dy*dy
                dB.append((dist_sq, s))
            dB.sort(key=lambda x: x[0])
            nearest_8B = [info[1] for info in dB[:8]] if len(dB)>=8 else [info[1] for info in dB]

            if filter_center(centerB, nearest_8B):
                faceB = [centerB] + nearest_8B
                grouped_faces.append(faceB)
            
            if len(grouped_faces) == 2:
                cx1, cy1 = centerA['cx'], centerA['cy']
                cx2, cy2 = centerB['cx'], centerB['cy']

                dist = np.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)
    
                # Dynamic threshold: Use avg spacing of nearby stickers to determine a valid min distance
                avg_sticker_size = np.mean([centerA['bbox'][2] - centerA['bbox'][0], 
                                            centerB['bbox'][2] - centerB['bbox'][0]])  # Avg width
                min_dist = avg_sticker_size * 1.5  # Ensure at least this much spacing
                
                if not dist >= min_dist:
                    pass
            else:
                nearest_8A_new = check_overlap(nearest_8A, nearest_8B)
                if len(nearest_8A) != len(nearest_8A_new):
                    pass
        else:
            pass
        

        face_colors = [
            (0,255,0),   # green
            (255,0,0),   # blue
            (0,255,255), # yellowish
            (255,0,255)  # magenta
        ]
        for f_idx, face_stickers in enumerate(grouped_faces):
            color = face_colors[f_idx % len(face_colors)]
            for st in face_stickers:
                # bounding box
                (x1, y1, x2, y2) = st['bbox']
                cls_id = st['class_id']
                if cls_id == 1:
                    # center => red
                    rect_color = (0,0,255)
                    thickness = 3
                    label_str = classify_center_color(frame, st['bbox'])
                else:
                    # normal sticker => face color
                    rect_color = color
                    thickness = 2
                    label_str = f"{class_names.get(cls_id)}"

                # draw
                cv2.rectangle(frame, (x1,y1), (x2,y2), rect_color, thickness)
                
                cv2.putText(frame, label_str, (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, rect_color, 2)
        
        # Display class counts on screen
        cv2.putText(frame, f"Faces: {len(grouped_faces)}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        # Show output frame
        cv2.imshow("YOLOv8 Live Detection", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def filter_center(center, normal_stickers, threshold_ratio = 0.5):
    if len(normal_stickers) < 8:
        return False
    cx, cy = center['cx'], center['cy']
    distances = []
    for sticker in normal_stickers:
        sx, sy = sticker['cx'], sticker['cy']
        dist = np.sqrt((sx - cx) ** 2 + (sy - cy) ** 2)  # Euclidean distance
        distances.append(dist)

    # Compute the mean distance and variance
    mean_dist = np.mean(distances)
    max_dev = max(abs(d - mean_dist) for d in distances)

    # Allowable variation in distances (set as a percentage of mean distance)
    tolerance = mean_dist * threshold_ratio

    return max_dev <= tolerance
        

def print_cube_net(face_data):
    """
    Print an ASCII net of the Rubik's cube in the standard orientation:
       White = top
       Green = left
       Blue = front
       Red = right
       Orange = back
       Yellow = bottom

    face_data[face_name] should be a 3x3 list of color labels, e.g.:
      face_data["White"] = [
        ["W","W","W"],
        ["W","W","W"],
        ["W","W","W"]
      ]
    """

    # For readability, let’s store them in variables
    white   = face_data["white"]   # 3x3
    yellow  = face_data["yellow"]  # 3x3
    blue    = face_data["blue"]    # 3x3
    green   = face_data["green"]   # 3x3
    red     = face_data["red"]     # 3x3
    orange  = face_data["orange"]  # 3x3

    # Helper to join row items, e.g. "W W W"
    def row_str(row):
        return " ".join(row)

    print("\n-- Rubik's Cube Net Layout --\n")

    # 1) Print White (top) - 3 lines, centered
    #   We'll indent with ~8 spaces for a "centered" look
    for r in range(3):
        print("        " + row_str(white[r]))

    # 2) Print Green, Blue, Red, Orange in a row
    for r in range(3):
        left_row   = row_str(red[r])
        front_row  = row_str(blue[r])
        right_row  = row_str(orange[r])
        back_row   = row_str(green[r])
        print("{}   {}   {}   {}".format(left_row, front_row, right_row, back_row))

    # 3) Print Yellow (bottom)
    for r in range(3):
        print("        " + row_str(yellow[r]))

    print("\n--------------------------------\n")

def classify_center_color(frame, bbox):
    """
    Given the bounding box of the center (x1,y1,x2,y2),
    crop that region from `frame`, and run a color classification.
    Returns one of the color strings: 'blue','red','green','orange','white','yellow'.
    """
    (x1, y1, x2, y2) = bbox
    cropped = frame[y1:y2, x1:x2]  # OpenCV format = [rowMin:rowMax, colMin:colMax]
    color_ranges = {
    "red": [
        ((0, 100, 50), (10, 255, 255)),       # "red1"
        ((170, 100, 50), (180, 255, 255))     # "red2"
    ],
    "orange": [
        ((10, 120, 50), (25, 255, 255))
    ],
    "green": [
        ((35, 50, 50), (90, 255, 255))
    ],
    "blue": [
        ((90, 100, 50), (140, 255, 255))      # Only 1 subrange, but in a list
    ],
    "yellow": [
        ((25, 150, 150), (40, 255, 255))
    ],
    "white": [
        ((0, 0, 200), (180, 20, 240))
    ]
}
    # Example: a quick HSV mean approach
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    color_counts = {}
    for color_name, ranges in color_ranges.items():
        total_count = 0
        for (lower, upper) in ranges:
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)

            mask = cv2.inRange(hsv, lower_np, upper_np)  # 255 where in range
            count = cv2.countNonZero(mask)
            total_count += count

        color_counts[color_name] = total_count

    # 4) Pick the color with the largest count
    best_color = max(color_counts, key=color_counts.get)
    return best_color

def draw_cube_net_image(face_data):
    """
    face_data: a dict with 3x3 arrays for each of these keys:
       "White", "Yellow", "Blue", "Green", "Red", "Orange"
    Returns an image (OpenCV BGR) showing the net layout.

    Net layout (face_width = 90 if square_size=30):
    
              [White]
    [Green] [Blue] [Red] [Orange]
              [Yellow]
    """

    square_size = 30
    face_width  = 3 * square_size
    face_height = 3 * square_size

    # We'll compute total width/height needed
    # The widest row is 4 faces wide => 4 * face_width = 360
    # The total height is 3 faces tall => 3 * face_width = 270
    # plus 1 face offset on top for White
    # Actually, let's just do 270 in height, 360 in width is enough
    net_height = 3 * face_width  # 270
    net_width  = 4 * face_width  # 360

    # Create a blank BGR image
    net_img = np.zeros((net_height, net_width, 3), dtype=np.uint8)

    def get_3x3_colors(face_name):
        face_3x3 = face_data.get(face_name)
        if face_3x3 is None:
            # return a 3x3 of placeholder label, say "X"
            return [
                ["X","X","X"],
                ["X","X","X"],
                ["X","X","X"]
            ]
        else:
            return face_3x3
    # define a small function to draw a 3x3 face at a given top-left
    def draw_face(face_name, top_left_x, top_left_y):
        face_3x3 = get_3x3_colors(face_name)  # e.g. [ ["B","B","B"], ["B","B","B"], ["B","B","B"] ]
        for row in range(3):
            for col in range(3):
                color_label = face_3x3[row][col]
                bgr = COLOR_MAP.get(color_label, (128,128,128))  # default gray if unknown
                # compute rectangle coords
                rect_x = top_left_x + col * square_size
                rect_y = top_left_y + row * square_size
                cv2.rectangle(
                    net_img,
                    (rect_x, rect_y),
                    (rect_x + square_size, rect_y + square_size),
                    bgr, 
                    thickness=-1   # filled
                )
                # optional black border
                cv2.rectangle(
                    net_img,
                    (rect_x, rect_y),
                    (rect_x + square_size, rect_y + square_size),
                    (0, 0, 0), 1
                )

    # Now we place each face at the correct position
    # White face: x=face_width (90), y=0
    draw_face("white",  face_width, 0)

    # Green, Blue, Red, Orange row => y=face_width (90)
    draw_face("red",  0,         face_width)
    draw_face("blue",   face_width,face_width)
    draw_face("orange",    2*face_width, face_width)
    draw_face("green", 3*face_width, face_width)

    # Yellow face: x=face_width (90), y=2*face_width (180)
    draw_face("yellow", face_width, 2*face_width)

    return net_img

def overlay(annotated_frame, net_img, scale = 0.5):
    combined = annotated_frame.copy()
    
    # 2) Resize the net image
    net_h, net_w = net_img.shape[:2]
    new_w = int(net_w * scale)
    new_h = int(net_h * scale)
    net_resized = cv2.resize(net_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 3) Compute the bottom-right corner coordinates
    h, w = combined.shape[:2]
    x_start = w - new_w
    y_start = h - new_h
    
    # 4) If the net is bigger than the frame in either dimension, handle partial or skip
    if x_start < 0 or y_start < 0:
        # net is too large for the given scale to fit in bottom-right
        # Optionally just place it at (0,0), or skip
        x_start = max(x_start, 0)
        y_start = max(y_start, 0)
    
    # 5) Overlay
    roi = combined[y_start:y_start+new_h, x_start:x_start+new_w]
    roi[:] = net_resized
    
    return combined

def compare():
    # test_images = ["test_dataset/images/IMG_9051.jpg"]
    # test_images = ["test_dataset/images/IMG_9051.jpg", "test_dataset/images/IMG_9055.jpg"]
    test_folder = "cube-vision/yolo_sticker/test/images"
    test_images = [os.path.join(test_folder, f) for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.png'))]
    # Load your trained YOLOv8 model
    yolo_model = YOLO("cube-vision/runs/detect/train19/weights/best.pt")
    # yolo_model = YOLO("runs/detect/train/weights/best.pt")
    # Load the Rubiks camSolver model
    rc_model = YOLO("RubikCubeColorExtractor-main/cube_detector.pt")
    for image_path in test_images:
        image_name = os.path.basename(image_path)
        ground_truth = load_ground_truth(image_path)
        if not ground_truth:
            print(f"No ground truth found for")

        # YOLOv8
        yolo_detections, yolo_time = test_yolo(image_path, yolo_model)
        evaluate_model("YOLOv8", yolo_detections, ground_truth, False, image_name, yolo_time)

        # Rubiks Camsolver
        rc_detections, rc_time = test_rc(image_path, rc_model)
        # print(rc_detections)
        evaluate_model("Rubiks CamSolver", rc_detections, ground_truth, True, image_name, rc_time)

        visualize_detections(image_path, ground_truth, yolo_detections, rc_detections)
        # print(f"YOLOv8 Inference Time: {yolo_time:.4f}s")
        # print(f"CamSolver Inference Time: {rc_time:.4f}s")
        # Print summary as a table
    print("\n **Model Performance Summary:**")
    print(tabulate(results_summary, headers=["Image", "Model", "Precision", "Recall", "F1 Score", "Inference Time"], tablefmt="fancy_grid"))

def cam():
    yolo_model = YOLO("cube-vision/runs/detect/train15/weights/best.pt")
    cap = cv2.VideoCapture(0)  # Use 0 if DroidCam is set as the default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break


        results = yolo_model.predict(frame, conf=0.5)

        annotated_frame = frame.copy()
        if len(results) > 0:
            boxes = results[0].boxes  # ultralytics 'Boxes' object
            for box in boxes:
                # box.xyxy[0] = [x1, y1, x2, y2] in floats
                x1, y1, x2, y2 = box.xyxy[0]
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                # Optionally get class name if your model has .names
                class_name = results[0].names[class_id] if hasattr(results[0], 'names') else str(class_id)

                # Convert to integer coords for drawing
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Put label text
                cv2.putText(annotated_frame, class_name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('Phone Webcam', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def detect_six_faces():
    yolo_model = YOLO("cube-vision/runs/detect/train15/weights/best.pt")

    face_order = [
        {"face_name": "blue",   "prompt": "Please show the Blue-centered face with White on top."},
        {"face_name": "red",    "prompt": "Now show the Red-centered face with White on top."},
        {"face_name": "green",  "prompt": "Now show the Green-centered face with White on top."},
        {"face_name": "orange", "prompt": "Now show the Orange-centered face with White on top."},
        {"face_name": "white",  "prompt": "Now show the White-centereqreen on top."},
        {"face_name": "yellow", "prompt": "Finally, show the Yellow-centered face Blue on top."}
    ]

    face_data = {
        "white": None,
        "yellow": None,
        "blue": None,
        "green": None,
        "red": None,
        "orange": None
    }
    cap = cv2.VideoCapture(0)  # Use 0 if DroidCam is set as the default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    for face_info in face_order:
        face_name = face_info["face_name"]
        prompt = face_info["prompt"]
        face_3x3_final = []
        face_detected = False

        stable_count = 0
        last_center = None
        while not face_detected:
            ret, frame = cap.read()
            if not ret:
                print("Error: could not read frame")
                break

            success, face_3x3 = detect_one_face(frame, yolo_model)
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, prompt, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            net_img = draw_cube_net_image(face_data)
            combined_frame = overlay(annotated_frame, net_img)
            cv2.imshow("Rubik Face Detection", combined_frame)
            # cv2.imshow("Rubik Face Detection", annotated_frame)

            if success:
                center_color = face_3x3[1][1]
                if center_color == face_name:
                    stable_count += 1
                else:
                    stable_count = 0
                last_center = center_color
                if stable_count >= 5:
                    face_3x3_final = face_3x3
                    face_detected = True
                    print(f"{face_name} Face detected")
            else:
                stable_count = 0
                last_center = None
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if face_detected:
            # Store the 3x3 layout in our dictionary
            face_data[face_name] = face_3x3_final
            print(f"{face_name} detected: {face_3x3_final}")
        else:
            # If we broke early or couldn't detect, we skip the rest
            print(f"Could not detect {face_name} face. Stopping.")
            break
    print("All face detection done. Press 'q' to exit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # We can show the final net in the corner
        net_img = draw_cube_net_image(face_data)
        combined_frame = overlay(frame.copy(), net_img)  # put the net in corner
        cv2.imshow("Rubik Face Detection", combined_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    return face_data

def bulid_face_3x3_with_clustering(projected):
    u2_vals = np.array([p['u2'] for p in projected]).reshape(-1,1)

    # 2) KMeans with 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42).fit(u2_vals)
    row_labels = kmeans.labels_  # array of size 9, each in {0,1,2}

    # 3) Group by row label
    row_dict = {0: [], 1: [], 2: []}
    for i, item in enumerate(projected):
        label = row_labels[i]
        row_dict[label].append(item)

    # 4) Sort row clusters by the mean 'u2' to get top->bottom
    label_order = []
    for label in [0,1,2]:
        cluster_u2 = [it['u2'] for it in row_dict[label]]
        mean_u2 = np.mean(cluster_u2)
        label_order.append((mean_u2, label))
    label_order.sort(key=lambda x: x[0])  # sort by mean u2 ascending

    # 5) Build final 3x3
    face_3x3 = []
    for _, row_label in label_order:
        row_items = row_dict[row_label]
        # within that row, sort by u1 ascending for left->right
        row_sorted = sorted(row_items, key=lambda d: d['u1'])
        face_3x3.append(row_sorted)

    return face_3x3

def check_face_equivalent(face, gt):
    t0 = gt
    t1 = rotate_90(gt)
    t2 = rotate_180(gt)
    t3 = rotate_270(gt)

    # Flip horizontally each rotation
    f0 = flip_horizontal(t0)
    f1 = flip_horizontal(t1)
    f2 = flip_horizontal(t2)
    f3 = flip_horizontal(t3)
    candidates = [t0, t1, t2, t3, f0, f1, f2, f3]
    for cand in candidates:
        if cand == face:
            return True
    return False

def rotate_matrix(theta):
    """
    Builds a 2x2 rotation matrix for angle -theta (clockwise rotation).
    Usually rotation by +theta is
      [ cosθ  -sinθ ]
      [ sinθ   cosθ ]
    so to rotate by -theta, we can just use cos(-θ) = cos(θ), sin(-θ) = -sin(θ).
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    # rotate by -theta
    R = np.array([
        [ cos_t,  sin_t],
        [-sin_t,  cos_t]
    ], dtype=np.float32)
    return R

def map_to_3x3(face_stickers, center):
    # print(face_stickers)
    center_cx = center['cx']
    center_cy = center['cy']
    full_stickers = [center] + face_stickers
    relative_pos = []
    for sticker in full_stickers:
        dx = sticker['cx'] - center_cx
        dy = sticker['cy'] - center_cy
        relative_pos.append([dx, dy])

    pca = PCA(n_components=2)
    pca.fit(relative_pos)
    pc1, pc2 = pca.components_

    projections = []
    for dx, dy in relative_pos:
        proj1 = dx * pc1[0] + dy * pc1[1]  # PC1 projection
        proj2 = dx * pc2[0] + dy * pc2[1]  # PC2 projection
        projections.append((proj1, proj2))
    
    # Cluster rows using PC2 projections (k=3 for 3 rows)
    row_proj = np.array([p[1] for p in projections]).reshape(-1, 1)
    kmeans = KMeans(n_clusters=3, n_init=100).fit(row_proj)
    row_labels = kmeans.labels_

    row_dict = {0: [], 1:[], 2:[]}
    for i, label in enumerate(row_labels):
        row_dict[label].append(i)
    row_mean = []
    for rlabel in [0,1,2]:
        if len(row_dict[rlabel]) == 0:
            mean_val = 9999999  # if empty
        else:
            these_proj2 = [projections[idx][1] for idx in row_dict[rlabel]]
            mean_val = np.mean(these_proj2)
        row_mean.append((mean_val, rlabel))
    row_mean.sort(key=lambda x: x[0])  # ascending => top->bottom

    # 6) Build a 3x3 structure
    face_3x3 = [[None, None, None],
                [None, None, None],
                [None, None, None]]

    # We'll place each cluster in a row => row0, row1, row2 in final
    # then we sort that row by proj1 => left->right
    for row_i, (_, cluster_label) in enumerate(row_mean):
        # row_indices = all stickers in that cluster
        row_indices = row_dict[cluster_label]
        # sort them by proj1
        row_stickers = []
        for idx in row_indices:
            p1 = projections[idx][0]
            row_stickers.append( (p1, full_stickers[idx]) )
        row_stickers.sort(key=lambda x: x[0])  # ascending by p1

        # Now we place them in face_3x3[row_i][col_i]
        # if we have exactly 3, it fits perfectly
        # if more or fewer, we handle best we can
        for col_i in range(min(3, len(row_stickers))):
            face_3x3[row_i][col_i] = row_stickers[col_i][1]

    return face_3x3

def robust_one_face(face_stickers):
    """
    face_sticker: list of dict, with cx, cy, bbox. len should be 9(center + 8 others)
    return 3x3 matrix represneting top to bottom, left to right ordering of the 9 stickers
    """
    if len(face_stickers) != 9:
        raise ValueError("Expect exactly 9 stickers for one face")
    
    points = []
    for s in face_stickers:
        points.append([s['cx'], s['cy']])
    points = np.array(points, dtype=np.float32)

    mean_x = np.mean(points[:,0])
    mean_y = np.mean(points[:,1])
    centered_points = points - [mean_x, mean_y]

    U, S, Vt = np.linalg.svd(centered_points, full_matrices=False)
    v1 = Vt[0]
    v2 = Vt[1]

    theta = np.arctan2(v1[1], v1[0])
    R = rotate_matrix(theta)

    # if S[0] < S[1]:
    #     S[0], S[1] = S[1], S[0]
    #     temp = v1.copy()
    #     v1 = v2
    #     v2 = temp

    rotated = []
    projected = []
    for i, s in enumerate(face_stickers):
        cx = s['cx'] - mean_x
        cy = s['cy'] - mean_y
        u1 = cx*v1[0] + cy*v1[1]
        u2 = cx*v2[0] + cy*v2[1]
        vec = np.array([cx, cy], dtype=np.float32)
        rx, ry = R @ vec  # matrix multiply => shape(2,)
        rotated.append({
            'index': i,
            'rx': rx,
            'ry': ry,
            'data': s
        })
        projected.append({
            'index': i,
            'u1': u1,
            'u2': u2,
            'data': s
        })

        sorted_pts = sorted(rotated, key=lambda d: (d['ry'], d['rx']))
        row1 = sorted_pts[0:3]
        row2 = sorted_pts[3:6]
        row3 = sorted_pts[6:9]
        row1_sorted = sorted(row1, key=lambda d:d['rx'])
        row2_sorted = sorted(row2, key=lambda d:d['rx'])
        row3_sorted = sorted(row3, key=lambda d:d['rx'])
    
    # face_3x3 = bulid_face_3x3_with_clustering(projected)

    # all_u1 = [p['u1'] for p in projected]
    # all_u2 = [p['u2'] for p in projected]
    # u1_min, u1_max = min(all_u1), max(all_u1)
    # u2_min, u2_max = min(all_u2), max(all_u2)
    # def linspace3(vmin, vmax):
    #     # e.g. vmin + (vmax-vmin)* ( [0.1667, 0.5, 0.8333] ) or similar
    #     # or we can do equally spaced edges
    #     return [
    #         vmin + (vmax - vmin) * (1/6),
    #         vmin + (vmax - vmin) * (3/6),
    #         vmin + (vmax - vmin) * (5/6)
    #     ]

    # row_centers = linspace3(u2_min, u2_max)
    # col_centers = linspace3(u1_min, u1_max)

    # cost_matrix = np.zeros((9,9), dtype=np.float32)

    # for i, sticker in enumerate(projected):
    #     u1_i, u2_i = sticker['u1'], sticker['u2']
    #     for slot_j in range(9):
    #         r = slot_j // 3
    #         c = slot_j % 3
    #         # center of that slot
    #         slot_u1 = col_centers[c]
    #         slot_u2 = row_centers[r]
    #         # Euclidean dist or squared dist
    #         dx = u1_i - slot_u1
    #         dy = u2_i - slot_u2
    #         dist = (dx*dx + dy*dy)**0.5  # sqrt
    #         cost_matrix[i, slot_j] = dist
    # # 6) Hungarian algorithm
    # row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # face_3x3 = [[None for _ in range(3)] for _ in range(3)]
    # for i in range(9):
    #     slot_j = col_ind[i]
    #     r = slot_j // 3
    #     c = slot_j % 3
    #     face_3x3[r][c] = projected[i]  # store the dict with 'data','u1','u2'

    face_3x3 = [row1_sorted, row2_sorted, row3_sorted]
    return face_3x3

def robust_mapping():
    yolo_model = YOLO("cube-vision/runs/detect/train19/weights/best.pt")
    face_data = {
        "white": None,
        "yellow": None,
        "blue": None,
        "green": None,
        "red": None,
        "orange": None
    }
    face_gt = [
            ['blue', 'white', 'yellow'],
            ['green', 'center', 'orange'],
            ['white', 'yellow', 'green']
            ]
    cap = cv2.VideoCapture(0)  # Use 0 if DroidCam is set as the default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    total_count = 0
    correct = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: could not read frame")
            break
        total_count += 1
        results = yolo_model.predict(source=frame, conf=0.5, save=False, verbose=False)
        centers = []
        normal_stickers = []

        if len(results) == 0:
            return None
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if class_id == 1:
                centers.append({'class_id': class_id, 'bbox': (x1,y1,x2,y2), 'cx': cx, 'cy': cy})
            else:
                normal_stickers.append({'class_id': class_id, 'bbox': (x1,y1,x2,y2), 'cx': cx, 'cy': cy})
        normal_stickers = check_overlap(normal_stickers, centers)
        grouped_faces = []
        if len(centers) == 1: # There is one face detected
            center_det = centers[0]

            distances = []
            for s in normal_stickers:
                dx = s['cx'] - center_det['cx']
                dy = s['cy'] - center_det['cy']
                dist = dx*dx + dy+dy
                distances.append((dist, s))
            distances.sort(key=lambda d: d[0])

            if len(distances) >= 8:
                nearest_8 = [info[1] for info in distances[:8]]
                faceA = [center_det] + nearest_8
                grouped_faces.append(faceA)


                # do sorting stuff, could move to later part
                face_3x3 = map_to_3x3(nearest_8, center_det)
                # face_3x3 = robust_one_face(face_sticker)
                # print(face_3x3)
                print("Detected One Face 3x3 Layout")
                face_3x3_matrix = []
                for row_i, row in enumerate(face_3x3):
                    row_labels = []
                    for col_i, sticker_dict in enumerate(row):
                        if sticker_dict is None:
                            break
                        # sticker_dict = item['data']
                        cls_id = sticker_dict.get('class_id')
                        if (cls_id == 1): # center sticker
                            bbox = sticker_dict['bbox']
                            # color = classify_center_color(frame, bbox)
                            row_labels.append("center")
                        else:
                            row_labels.append(class_names.get(cls_id))
                    face_3x3_matrix.append(row_labels)
                    # print(f"Row {row_i}: {row_labels}")
                
                if check_face_equivalent(face_3x3_matrix, face_gt):
                    print("True")
                    correct += 1
                else:
                    print("False")
                    print(face_3x3_matrix)
        elif len(centers) == 2:
            # Face A
            centerA = centers[0]
            dA = []
            for s in normal_stickers:
                dx = s['cx'] - centerA['cx']
                dy = s['cy'] - centerA['cy']
                dist_sq = dx*dx + dy*dy
                dA.append((dist_sq, s))
            dA.sort(key=lambda x: x[0])
            nearest_8A = [info[1] for info in dA[:8]] if len(dA)>=8 else [info[1] for info in dA]

            # Face B
            centerB = centers[1]
            dB = []
            for s in normal_stickers:
                dx = s['cx'] - centerB['cx']
                dy = s['cy'] - centerB['cy']
                dist_sq = dx*dx + dy*dy
                dB.append((dist_sq, s))
            dB.sort(key=lambda x: x[0])
            nearest_8B = [info[1] for info in dB[:8]] if len(dB)>=8 else [info[1] for info in dB]

            faceA = [centerA] + nearest_8A
            faceB = [centerB] + nearest_8B
            grouped_faces.append(faceA)
            grouped_faces.append(faceB)
        else:
            pass
        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(f"Correct Percentage:{correct / total_count}")
    cap.release()
    cv2.destroyAllWindows()



def main():
    face_data = detect_six_faces()
    if len(face_data) == 6:
        print("\nAll six faces were successfully detected!")
        net_img = draw_cube_net_image(face_data)
        # cv2.imshow("cube Net", net_img)
        print_cube_net(face_data)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # do something with face_data, e.g. solve the cube
    else:
        print(f"\nOnly {len(face_data)} faces were captured. Please try again.")


if __name__ == '__main__':
    # compare()
    # cam()
    # show_detection()
    main() 
    # robust_mapping()