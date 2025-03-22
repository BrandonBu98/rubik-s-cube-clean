import cv2
import os
from ultralytics import YOLO
import numpy as np
import supervision as sv
import time
from tabulate import tabulate

model = YOLO('cube-vision/best.pt')
results_summary = []

def detect_cube_and_crop(frame, model):
    # Run YOLO on the frame
    results = model(frame, verbose=False, iou=0.45)[0]
    # Convert YOLO results to Supervision's Detections
    detections = sv.Detections.from_ultralytics(results)

    # Assuming your model is trained to detect a single 'cube' class
    # or you want the bounding box with highest confidence:
    if len(detections) == 0:
        # No detection found, return None or the full frame
        return None  

    # E.g. pick the detection with the highest confidence
    best_idx = np.argmax(detections.confidence)
    x1, y1, x2, y2 = detections.xyxy[best_idx].astype(int)

    # Crop the region from the original frame
    # Make sure bounding box coords are in valid range
    h, w, _ = frame.shape
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)

    roi = frame[y1:y2, x1:x2].copy()

    return roi, (x1, y1, x2, y2)

def process_frame_with_roi(image, model):
    roi_result = detect_cube_and_crop(image, model)
    if not roi_result:
        print("No Rubik's cube detected!")
        # cv2.imshow("Output", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return image

    roi, (x1, y1, x2, y2) = roi_result
    # mask out the bound area
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    # cv2.imshow("masked Image", masked_image)
    return masked_image

def smooth_filter(image, mask):
    open_kernel =  np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    # cv2.imshow('closing', closing)
    close_kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, close_kernel)
    # cv2.imshow('opening', opening)

    # blob_image = np.zeros_like(opening)
    # for val in np.unique(opening)[1:]:
    #     mask = np.uint8(opening == val)
    #     labels, stats = cv2.connectedComponentsWithStats(opening, 4)[1:3]
    #     largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    #     # blob_image[labels == largest_label] = val
    #     blob_image[labels > 0] = 255
    # # cv2.imshow("Largest Region - Blob Image", blob_image)
    # final_output = cv2.bitwise_and(image, image, mask=blob_image)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closing, connectivity=4)
    filtered_mask = np.zeros_like(closing)
    # keep only bolbs above a certain area threshold
    min_area = 200   # you can tune
    for i in range(1, num_labels):  # label=0 is background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            filtered_mask[labels == i] = 255
    final_output = cv2.bitwise_and(image, image, mask=filtered_mask)
    return final_output, filtered_mask

def edge_detection(image, mask):
    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive threshold to emphasize stickers
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological operations to clean up noise
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    return morph

    # kernel =  np.ones((5,5),np.uint8)
    # dialated_mask = cv2.dilate(mask,kernel,iterations = 1)

    # edges = cv2.Canny(image,200,200)
    # edges = cv2.bitwise_and(edges, edges, mask=dialated_mask)
    # return edges

def detect_skin_hsv(bgr_image):
    """
    Returns a binary mask of likely skin pixels.
    """
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    # Example ranges for light/medium skin, these will need adjustment
    lower_skin = np.array([10, 40, 60], dtype=np.uint8)
    upper_skin = np.array([25, 120, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # ycrcb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCrCb)
    # # Example ranges for light/medium skin, these will need adjustment
    # lower_skin = np.array([0, 133, 77], dtype=np.uint8)  
    # upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    # skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    return skin_mask

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

def evaluate_model(model_name, detections, ground_truth, rc, image_name, inference_time):
    total_correct = 0
    total_detections = len(detections)
    total_ground_truth = len(ground_truth)
    print(f"detection is {detections}")

    for det in detections:
        class_id, det_shape = det
        matching_gt = [(gt_class, gt_polygon, gt_box) for gt_class, gt_polygon, gt_box in ground_truth if gt_class == class_id]
        best_iou = 0.0
        for (gt_class, gt_polygon, gt_box) in matching_gt:
            if rc:
                pass
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

def process_image(image):

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    masks = {}
    outputs = {}
    combined_mask = np.zeros_like(hsv_image[:, :, 0])

    # original hsv range
    # color_ranges = {
    #     "red": ((0, 190, 80), (5, 255, 255)),
    #     "orange": ((6, 210, 70), (20, 255, 255)),
    #     "green": ((50, 50, 50), (80, 255, 255)),
    #     "blue": ((100, 150, 0), (140, 255, 255)),
    #     "yellow": ((25, 150, 150), (35, 255, 255)),
    #     "white": ((0, 0, 200), (180, 20, 255))
    # }

    color_ranges = {
        "red1": ((0, 100, 50), (10, 255, 255)),   # First range of red
        "red2": ((170, 100, 50), (180, 255, 255)), # Second range of red
        "orange": ((10, 120, 50), (25, 255, 255)), 
        "green": ((35, 50, 50), (90, 255, 255)),  # Manually adjusted green
        "blue": ((90, 100, 50), (140, 255, 255)), # Manually adjusted blue
        "yellow": ((25, 150, 150), (40, 255, 255)), 
        "white": ((0, 0, 200), (180, 20, 240))  
    }

    for color, (lower, upper) in color_ranges.items():
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        # Create a mask for the current color range
        mask = cv2.inRange(hsv_image, lower, upper)
        masks[color] = mask  # Store mask
        combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Create an output image that shows the original image where the mask is white
        output = cv2.bitwise_and(image, image, mask=mask)
        outputs[color] = output  # Store output


        # cv2.imshow(f"{color} mask", mask)
        # cv2.imshow(f"{color} output", output)


    combined_output = np.zeros_like(image)
    # Combine all output images
    for color, output in outputs.items():
        # Where output is not black (i.e., the color is detected), copy it to the combined_output image
        combined_output[output > 0] = output[output > 0]

    for color, mask in masks.items():
        combined_mask[mask > 0] = mask[mask > 0]

    # cv2.imshow('combined mask', combined_mask)



    # Cleanup filtering
    skin_mask = detect_skin_hsv(combined_output)
    # cv2.imshow('skin mask', skin_mask)
    smoothed_mask = cv2.bitwise_and(combined_mask, cv2.bitwise_not(skin_mask))
    final_output, smoothed_mask = smooth_filter(combined_output, smoothed_mask)

    # final_output, smoothed_mask = smooth_filter(combined_output, combined_mask)
    edges = edge_detection(image, smoothed_mask)

    contours, _ = cv2.findContours(smoothed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(f"Found {len(contours)} contours in the mask.")
    valid_sticker = detect_stickers_from_contours(contours)

    # visualize_sticker_boxes(image, valid_sticker)

    # extract each sticker's color
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sticker_info = []
    if valid_sticker is not None:

        # ONLY FOR UPRIGHT TEST
        # for row in valid_sticker:
        #     row_info = []
        #     for box in row:
        #         color_name = get_sticker_color(hsv_frame, box, color_ranges)
        #         if (color_name) is None:
        #             row_info.append("None")
        #         else:
        #             row_info.append(color_name)
        #     sticker_info.append(row_info)
        # print(sticker_info)

        for row in valid_sticker:
            row_color = []
            for (cx, cy, corners) in row:
                # corners is a 4x2 array of points
                # if you want to do color detection:
                
                # Option A: fill a poly mask
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [corners], 255)
                
                hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                pixels = hsv_frame[mask == 255]
                mean_h = np.mean(pixels[:,0])
                mean_s = np.mean(pixels[:,1])
                mean_v = np.mean(pixels[:,2])

                color_name = None
                for cname, (lower, upper) in color_ranges.items():
                    (h_low, s_low, v_low) = lower
                    (h_high, s_high, v_high) = upper

                    if (h_low <= mean_h <= h_high and
                        s_low <= mean_s <= s_high and
                        v_low <= mean_v <= v_high):
                        color_name = cname
                        break
                row_color.append(color_name)
            sticker_info.append(row_color)
    contour_image = np.zeros_like(image)
    cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 3)
    kernel =  np.ones((3,3),np.uint8)
    contour_image = cv2.dilate(contour_image,kernel,iterations = 2)
    contour_image = cv2.morphologyEx(contour_image, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('contours', contour_image)

    tiles = cv2.bitwise_and(final_output, 255 - contour_image)
    # cv2.imshow('tiles', tiles)

    edges2 = cv2.Canny(tiles,30,200)
    # cv2.imshow('edge2', edges2)

    # Show the combined output image
    # cv2.imshow('Final Output', final_output)
    # cv2.imshow('Final mask', smoothed_mask)
    # cv2.imshow('edges', edges)
    return final_output, smoothed_mask, edges, contour_image, tiles, valid_sticker, sticker_info
    # return smoothed_mask

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

def get_sticker_color(hsv_frame, bbox, color_ranges):
    x, y, w, h = bbox

    # 1) Extract the subregion for this sticker
    sticker_roi = hsv_frame[y:y+h, x:x+w]
    if sticker_roi.size == 0:
        return None  # no pixels, maybe bounding box is out of range

    # 2) Compute the mean hue/sat/val
    mean_h = np.mean(sticker_roi[:, :, 0])
    mean_s = np.mean(sticker_roi[:, :, 1])
    mean_v = np.mean(sticker_roi[:, :, 2])

    # 3) Compare against each defined color range
    for color_name, (lower, upper) in color_ranges.items():
        (h_low, s_low, v_low) = lower
        (h_high, s_high, v_high) = upper

        if (h_low <= mean_h <= h_high and
            s_low <= mean_s <= s_high and
            v_low <= mean_v <= v_high):
            return color_name

    # If no range was matched, return None or "unknown"
    return None


def detect_stickers_from_contours(contours):
    """
    Uses contours to identify cube stickers.
    Returns bounding boxes sorted in a 3x3 grid.
    """
    valid_stickers = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        # print("contour area:", area)
        # print(f"Contour {i}: area={area}")
        if area < 2000 or area > 50000:
            # print("  -> Skipped due to area filter.")
            # tweak these thresholds to your scenario
            continue

        # Approximate the contour
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

        if len(approx) == 4:
            rect = cv2.minAreaRect(cnt)
            box_points = cv2.boxPoints(rect)
            box_points = np.int32(box_points)
            cx = np.mean(box_points[:, 0])
            cy = np.mean(box_points[:, 1])
            valid_stickers.append((cx, cy, box_points))

        # ONLY FOR UPRIGHT TEST
        # Check if it has 4 corners
        # if len(approx) == 4:
        #     # bounding box shape check
        #     x, y, w, h = cv2.boundingRect(approx)
        #     aspect_ratio = w / float(h)
        #     if 0.75 < aspect_ratio < 1.33:  # allow some tolerance for near-square
        #         valid_stickers.append((x, y, w, h))
    # print(f"Found {len(valid_stickers)} valid sticker in the mask.")
    # Sort stickers row-wise to form a 3x3 grid
    # valid_stickers.sort(key=lambda box: (box[1], -box[0]))
    valid_stickers.sort(key=lambda box: (box[1], box[0]))
    
    if len(valid_stickers) >= 9:
        # Take the first 9 in sorted order, or apply more advanced grouping
        face_stickers = valid_stickers[:9]
        # Optionally reshape them into 3 rows of 3
        # face_grid = [face_stickers[i:i+3] for i in range(0, 9, 3)]
        face_grid = sort_stickers_3x3(face_stickers)
        # print("face grid", face_grid)
    else:
        face_grid = None
    return face_grid

def sort_stickers_3x3(sticker_boxes):
    # sort all boxes by y ascending
    # (lowest y means top row)
    sorted_by_y = sorted(sticker_boxes, key=lambda b: b[1])

    # slice them into rows of 3
    # top row = sorted_by_y[0..2], middle = [3..5], bottom = [6..8]
    row1 = sorted_by_y[0:3]
    row2 = sorted_by_y[3:6]
    row3 = sorted_by_y[6:9]

    # within each row, sort by x descending (right to left)
    row1_sorted = sorted(row1, key=lambda b: b[0], reverse=True)
    row2_sorted = sorted(row2, key=lambda b: b[0], reverse=True)
    row3_sorted = sorted(row3, key=lambda b: b[0], reverse=True)

    return [row1_sorted, row2_sorted, row3_sorted]

def visualize_sticker_boxes(image, sticker_boxes):
    """
    Draws the detected sticker bounding boxes on the image for debugging.
    """
    for row in sticker_boxes:
        for corners in row:
            cx, cy, corner = corners
            corners_array = np.array(corner, dtype=np.int32)
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green boxes
            cv2.polylines(
                image,
                [corners_array],
                isClosed=True,
                color=(0, 255, 0),
                thickness=2
            )
    # ONLY FOR UPRIGHT TEST
    # for row in sticker_boxes:
    #     for bbox in row:
    #         x, y, w, h = bbox
    #         # corners_array = np.array(corner, dtype=np.int32)
    #         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green boxes

def detect_cube_face(image):
    """
    Finds the largest contour in the image, assuming it's the Rubik's Cube face.
    Draws the four corner points on the original image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Adaptive threshold to emphasize cube stickers
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological operations to merge stickers into one contour
    kernel = np.ones((5,5), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    cv2.imshow("All Contours", contour_image)

    largest_contour = max(contours, key=cv2.contourArea)
    contour_image_max = image.copy()
    cv2.drawContours(contour_image_max, [largest_contour], -1, (0, 255, 0), 3)  # Green contour
    cv2.imshow("Largest Contours", contour_image_max)

    if not contours:
        print("No contours found!")
        return image  # Return original image if no contours found

    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)


    # Try to find a contour that represents the full cube face
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 10000:  # Ignore very small areas
            continue

        # Get convex hull to enclose all stickers
        hull = cv2.convexHull(contour)

        # Approximate the shape to reduce noise
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)

        if len(approx) == 4:  # We need exactly 4 corner points
            for point in approx:
                x, y = point[0]
                cv2.circle(image, (x, y), 8, (0, 0, 255), -1)  # Draw red points

            print("Detected Cube Face Corners:", approx.reshape(4, 2))
            return image  # Return image with drawn corners and coordinates

    print("Could not find the full cube face.")
    return image

def make_vid():
    # videos_list = ["test1.mp4","test2.mp4", "test3.mp4", "test4.mp4", "test5.mp4"]
    videos_list = ["test_slow.mp4"]
    input_folder = "data/"
    output_folder = "output/"

    os.makedirs(output_folder, exist_ok=True)
    for video_filename in videos_list:
        start_time = time.time()

        video_path = os.path.join(input_folder, video_filename)
        video_name = os.path.splitext(video_filename)[0]
        video_output_folder = os.path.join(output_folder, video_name)
        os.makedirs(video_output_folder, exist_ok=True)

        output_path = os.path.join(video_output_folder, f"output_{video_filename}")
        mask_path = os.path.join(video_output_folder, f"mask_{video_filename}")
        edges_path = os.path.join(video_output_folder, f"edges_{video_filename}")
        tiles_path = os.path.join(video_output_folder, f"tiles_{video_filename}")
        print(f"Processing: {video_filename}")

        cam = cv2.VideoCapture(video_path)
        ret, image = cam.read()
        if not ret or image is None:
            print(f"Error reading {video_filename}, skipping...")
            continue

        frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (frame_width, frame_height), True)
        mask_output = cv2.VideoWriter(mask_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (frame_width, frame_height), False)
        edges_output = cv2.VideoWriter(edges_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (frame_width, frame_height), False)
        tiles_output = cv2.VideoWriter(tiles_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (frame_width, frame_height), True)

        frame_count = 0
        while ret:
            # cv2.imshow('video', frame)
            masked_image = process_frame_with_roi(image, model)
            output_image, smoothed_mask, edges, _, tiles, valid_sticker, detected_colors = process_image(masked_image)
            # print(outputs.shape)
            annotated_frame = output_image.copy()
            # print("Detected colors:", detected_colors)
            frame_height, frame_width = annotated_frame.shape[:2]
            row_height = 50
            col_width = 200
            start_x, start_y = 700, 200
                
            if detected_colors and valid_sticker is not None:
                visualize_sticker_boxes(annotated_frame, valid_sticker)
                for row_idx, row in enumerate(valid_sticker):
                    for col_idx, corners in enumerate(row):
                        cx, cy, corner = corners
                        color_name = detected_colors[row_idx][col_idx]
                        if color_name is None:
                            color_name = "None"
                        corner = np.array(corner)
                        cx = int(cx)
                        cy = int(cy)
                        cv2.putText(
                            annotated_frame,
                            str(color_name),
                            (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,  # font scale
                            (0, 255, 0),  # text color (B,G,R)
                            2   # thickness
                        )

                # ONLY FOR UPRIGHT TEST
                # for row_idx, row in enumerate(valid_sticker):
                #     for col_idx, bbox in enumerate(row):
                #         x, y, w, h = bbox
                #         color_name = detected_colors[row_idx][col_idx]
                #         if color_name is None:
                #             color_name = "None"
                #         cx = x + w // 2
                #         cy = y + h // 2
                #         cv2.putText(
                #             annotated_frame,
                #             str(color_name),
                #             (cx, cy),
                #             cv2.FONT_HERSHEY_SIMPLEX,
                #             1.5,  # font scale
                #             (0, 255, 0),  # text color (B,G,R)
                #             2   # thickness
                #         )
            else:
                cv2.putText(
                    annotated_frame,
                    "No stickers found",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 255),
                    2
                )
                
            video_output.write(annotated_frame)
            mask_output.write(smoothed_mask)
            edges_output.write(edges)
            tiles_output.write(tiles)
            frame_count += 1
            ret, image = cam.read()
                
        # cv2.destroyAllWindows()
        video_output.release()
        mask_output.release()
        edges_output.release()
        tiles_output.release()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Finished processing {video_filename} - Total frames: {frame_count}")
        print(f"Processing time for {video_filename}: {elapsed_time:.2f} seconds")

def extract_frame(video_path, output_image_path):
    cam = cv2.VideoCapture(video_path)
    ret, frame = cam.read()  # Read the first frame
    if ret:
        cv2.imwrite(output_image_path, frame)  # Save frame as image
        print(f"Saved test frame as {output_image_path}")
    else:
        print("Error: Could not read video frame.")
    cam.release()

def test_image():
    # extract_frame("data/test1.mp4", "test_frame.png")
    # image = cv2.imread('test_frame.png')
    # if image is None:
    #     print("Error: Image could not be loaded.")
    #     return
    # output_cube_face = detect_cube_face(image)

    test_folder = "cube-vision/yolo_sticker/test/images"
    test_images = [os.path.join(test_folder, f) for f in os.listdir(test_folder) if f.lower().endswith(('.jpg', '.png'))]
    for image_path in test_images:
        image_name = os.path.basename(image_path)
        ground_truth = load_ground_truth(image_path)
        if not ground_truth:
            print(f"No ground truth found for")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            continue
    # cv2.imshow("Detected Cube Corners", output_cube_face)
    # Detect bounding box and get ROI
        masked_image = process_frame_with_roi(image, model)

        output_image, smoothed_mask, edges, contour_image, tiles, valid_sticker, detected_colors = process_image(masked_image)
        detection_results = []
        start_time = time.time()
        if valid_sticker is not None:
            for row in valid_sticker:
                for (cx, cy, corners) in row:
                    class_id = detected_colors[row.index((cx, cy, corners))]
                    detection_results.append((class_id, corners.tolist()))
        
        inference_time = time.time() - start_time
        evaluate_model("OpenCV", detection_results, ground_truth, False, image_name, inference_time)
    print(tabulate(results_summary, headers=["Image", "Model", "Precision", "Recall", "F1 Score", "Inference Time"], tablefmt="fancy_grid"))


    # cv2.imshow('Original Frame', image)
    # cv2.imshow('output image', output_image)
    # cv2.imshow('Smoothed Mask', smoothed_mask)
    # cv2.imshow('Edges', edges)
    # cv2.imshow('Contours', contour_image)
    # cv2.imshow('Extracted Tiles', tiles)
    # print(detected_colors)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def main():
    # List of test videos to process
    # test_videos = ["test1.mp4", "test2.mp4", "test3.mp4", "test4.mp4", "test5.mp4"]
    test_videos = ["test1.mp4"]
    video_folder = "data/"
    output_folder = "output/"
    model = YOLO('best.pt')
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    for video_filename in test_videos:
        video_path = os.path.join(video_folder, video_filename)
        output_path = os.path.join(output_folder, f"mask_{video_filename}")
        print(f"Processing: {video_filename}")
        cam = cv2.VideoCapture(video_path)
        ret, image = cam.read()

        if not ret:
            print("Error: Could not read video frame. Check file format or encoding.")
            return  # Exit the function if video cannot be read
        frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # video_output=cv2.VideoWriter('output/frame_mask1.mp4',cv2.VideoWriter_fourcc(*"mp4v"),30,(720, 1280), False)
        video_output = cv2.VideoWriter(output_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        30, (frame_width, frame_height), False)
        while ret:
            # outputs = process_image(frame)

            # image = cv2.imread('test-cube.png')
            masked_image = process_frame_with_roi(image, model)
            output_image, smoothed_mask, _, _, _, _, _ = process_image(masked_image)
            # cv2.imshow('smoothed mask', smoothed_mask)

            # Hard coded mask
            mask = np.zeros(image.shape[:2], dtype="uint8")
            height, width, _ = image.shape
            size = min(width, height) // 2
            # print(width, height)
            cv2.rectangle(mask, (width//2 - size, height//2 - size),(width//2 + size, height//2 + size), 255, -1)

            rect_area = cv2.countNonZero(mask)
            # cv2.imshow('mask', mask)

            output_mask = cv2.bitwise_and(smoothed_mask, mask)
            output_mask_inv = cv2.bitwise_not(mask)
            output_mask_inv = cv2.bitwise_and(smoothed_mask, output_mask_inv)
            # cv2.imshow('output mask', output_mask)
            # cv2.imshow('output mask inv', output_mask_inv)

            pixels = cv2.countNonZero(output_mask) / rect_area * 100
            pixels_inv = cv2.countNonZero(output_mask_inv) / rect_area * 100


            # print(pixels, pixels_inv, rect_area)
            if pixels > 25 and pixels_inv < 40:
                # print("Good frame")
                video_output.write(output_image)
            video_output.write(smoothed_mask)
            # ret = False
            ret, image = cam.read()
                
        video_output.release()

    # image = cv2.imread('test-cube.png') 
    # output_image, smoothed_mask, edges, _, _, _ = process_image(image)
    # cv2.imshow('smoothed mask', smoothed_mask)
    #
    # # Hard coded mask
    # mask = np.zeros(image.shape[:2], dtype="uint8")
    # width, height, _ = image.shape
    # cv2.rectangle(mask, (int(width * 0.2), int(height * 0.25)), (int(width * 0.7), int(height * 0.75)), 255, -1)
    #
    # rect_area = cv2.countNonZero(mask)
    # cv2.imshow('mask', mask)
    #
    # output_mask = cv2.bitwise_and(smoothed_mask, mask)
    # output_mask_inv = cv2.bitwise_not(mask)
    # output_mask_inv = cv2.bitwise_and(smoothed_mask, output_mask_inv)
    # cv2.imshow('output mask', output_mask)
    # cv2.imshow('output mask inv', output_mask_inv)
    #
    # pixels = cv2.countNonZero(output_mask) / rect_area * 100
    # pixels_inv = cv2.countNonZero(output_mask_inv) / rect_area * 100
    #
    #
    # print(pixels, pixels_inv, rect_area)
    # if pixels > 90 and pixels_inv < 10:
    #     print("Good frame")
    #
    #
    #
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def compare_image():
    yolo_model = YOLO("runs/detect/train15/weights/best.pt")

    image_path = "test_frame.png"
    image = cv2.imread(image_path)

    # YOLO Detection
    start_time = time.time()
    yolo_predictions = detect_yolo(image_path, yolo_model)
    end_time = time.time()
    yolo_time = end_time - start_time

    color_labels = ["blue", "green", "orange", "red", "white", "yellow"]
    for class_id, x1, y1, x2, y2 in yolo_predictions:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(image, f"YOLO: {color_labels[class_id]}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    
    # CV Detection
    cvimage_path = "test_frame.png"
    cvimage = cv2.imread(image_path)
    cv_model = YOLO("best.pt")
    start_time = time.time()
    cv_image,_ = detect_cv(cvimage, cv_model)
    end_time = time.time()
    cv_time = end_time - start_time

    combined_image = cv2.addWeighted(image, 0.6, cv_image, 0.4, 0)
    print("YOLO time is ", yolo_time, " seconds")
    print("Computer Vision time is ", cv_time, " seconds")
    cv2.imshow("sticker detection Comparison", combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def compare_vid():
    yolo_model = YOLO("runs/detect/train15/weights/best.pt")
    cv_model = YOLO("best.pt")

    start_time = time.time()
    videos_list = ["test_slow.mp4"]
    input_folder = "data/"
    output_folder = "output/compare/"
    os.makedirs(output_folder, exist_ok=True)
    for video_filename in videos_list:
        video_path = os.path.join(input_folder, video_filename)
        video_name = os.path.splitext(video_filename)[0]
        video_output_folder = os.path.join(output_folder, video_name)
        output_video_path = os.path.join(video_output_folder, "comparison_output.mp4")
        os.makedirs(video_output_folder, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width * 2, frame_height))

        frame_skip = 10
        frame_count = 0

        total_yolo_detections = 0
        correct_yolo_detections = 0
        total_cv_detections = 0
        correct_cv_detections = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                frame_yolo = frame.copy()
                frame_cv = frame.copy()
                # yolo_predictions = detect_yolo(frame_yolo, yolo_model)
                # frame_yolo = draw_predictions(frame_yolo, yolo_predictions, "YOLO")

                frame_cv, cv_predictions = detect_cv(frame_cv, cv_model)

                combined_frame = np.hstack((frame_yolo, frame_cv))
                out.write(combined_frame)

            frame_count += 1
    end_time = time.time()
    print("time taken: ", end_time - start_time)
    cap.release()
    out.release()

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    inter_x1 = max(x1, x1g)
    inter_y1 = max(y1, y1g)
    inter_x2 = min(x2, x2g)
    inter_y2 = min(y2, y2g)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def draw_predictions(frame, predictions, method="YOLO"):
    color_labels = ["blue", "green", "orange", "red", "white", "yellow"]
    colors = [(255, 0, 0), (0, 255, 0), (0, 165, 255), (0, 0, 255), (255, 255, 255), (0, 255, 255)]
    
    for class_id, x1, y1, x2, y2 in predictions:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colors[class_id], 2)
        cv2.putText(frame, f"{method}: {color_labels[class_id]}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, colors[class_id], 2, cv2.LINE_AA)
    
    return frame

def detect_yolo(image_path, model):
    yolo_results = model.predict(source=image_path, conf=0.4, save=False)

    yolo_predictions = []
    for result in yolo_results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Bounding box
            class_id = int(box.cls[0].cpu().numpy())  # Predicted class (color)
            yolo_predictions.append((class_id, x1, y1, x2, y2))
    return yolo_predictions

def detect_cv(image, model):
    masked_image = process_frame_with_roi(image, model)
    output_image, smoothed_mask, edges, _, tiles, valid_sticker, detected_colors = process_image(masked_image)
    annotated_frame = output_image.copy()
    # print("Detected colors:", detected_colors)
    frame_height, frame_width = annotated_frame.shape[:2]
    row_height = 50
    col_width = 200
    start_x, start_y = 700, 200
    
    cv_predictions = []
    if detected_colors and valid_sticker is not None:
        visualize_sticker_boxes(annotated_frame, valid_sticker)
        for row_idx, row in enumerate(valid_sticker):
            for col_idx, corners in enumerate(row):
                cx, cy, corner = corners
                color_name = detected_colors[row_idx][col_idx]
                if color_name is None:
                    color_name = "None"
                corner = np.array(corner)
                cx = int(cx)
                cy = int(cy)
                cv2.putText(
                    annotated_frame,
                    str(color_name),
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,  # font scale
                    (0, 255, 0),  # text color (B,G,R)
                    2   # thickness
                )
                cv_predictions.append((color_name, cx, cy))
    return annotated_frame, cv_predictions

if __name__ == '__main__':
    # make_vid()
    test_image()
    # main()
    # compare_vid()
    # compare_image()

