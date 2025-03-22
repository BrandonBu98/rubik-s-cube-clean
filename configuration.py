import time
import os
import cv2
import numpy as np
import math
from ultralytics import YOLO
from shapely.geometry import Polygon
from PIL import Image, ImageDraw
from IPython.display import display
from tabulate import tabulate
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull


class_names = {
    0: "blue",
    1: "center",
    2: "green",
    3: "orange",
    4: "red",
    5: "white",
    6: "yellow"
}

FACE_NEIGHBORS = {
    "blue": ["red", "white", "orange", "yellow"],
    "red": ["green", "white", "blue", "yellow"],
    "orange": ["blue", "white", "green", "yellow"],
    "green": ["orange", "white", "red", "yellow"],
    "white": ["red", "green", "orange", "blue"],
    "yellow": ["red", "blue", "orange", "green"]
}

corner_mapping = {
    "blue": {
        ("orange", "white"): 2,
        ("orange", "yellow"): 8,
        ("red", "white"): 0,
        ("red", "yellow"): 6
    },
    "red": {
        ("blue", "white"): 2,
        ("blue", "yellow"): 8,
        ("green", "white"): 0,
        ("green", "yellow"): 6
    },
    "green": {
        ("orange", "white"): 0,
        ("orange", "yellow"): 6,
        ("red", "white"): 2,
        ("red", "yellow"): 8
    },
    "orange": {
        ("blue", "white"): 0,
        ("blue", "yellow"): 6,
        ("green", "white"): 2,
        ("green", "yellow"): 8
    },
    "white": {
        ("blue", "orange"): 8,
        ("blue", "red"): 6,
        ("green", "orange"): 2,
        ("green", "red"): 0
    },
    "yellow": {
        ("blue", "orange"): 2,
        ("blue", "red"): 0,
        ("green", "orange"): 8,
        ("green", "red"): 6
    }
}

class CubeState:
    # face_id: center sticker color
    # center: class_id = 1, bbox, cx, cy
    # perimeter: list of 8 stickers sorted by angle around center
    def __init__(self):
        self.faces = {}
        self.confidence_threshold = 0.8
        self.net = {}
    
    def check_face(self, face_id, center, perimeter):
        ring = update_ring(center, perimeter)
        trust = True
        if face_id not in self.faces:
            self.faces[face_id] = {
                'center': center,
                'ring': ring,
                'ring_finalized': False,
                'confidence': 0.3,
                'stability': 0,
                'finalized': False,
                'neighbors': {
                    neighbor_color: None for neighbor_color in FACE_NEIGHBORS[face_id]
                }
            }
        else: # this face has been detected before
            stored_face = self.faces[face_id]
            similarity = compute_face_similarity(stored_face['ring'], ring)
            if self.faces[face_id]['ring_finalized'] == True:
                if similarity >= 0.9:
                    stored_face['confidence'] = min(1.0, stored_face['confidence'] + 0.1)
                    stored_face['stability'] += 1
                print(f"Face {face_id} is finalized, skip")
            else:
                if similarity >= 0.8:
                    stored_face['confidence'] = min(1.0, stored_face['confidence'] + 0.1)
                    stored_face['stability'] += 1
                    stored_face['ring'] = ring
                    stored_face['center'] = center
                    # print(f"confidence added FACE {face_id} with {stored_face['confidence']} and {stored_face['stability']}")
                    # print(", ".join(class_names.get(sticker['class_id'], "Unknown") for sticker in ring))
                else:
                    stored_face['stability'] = max(0, stored_face['stability'] - 1)
                    stored_face['confidence'] = max(0.0, stored_face['confidence'] - 0.05)
                    trust = False
                    # print(f"confidence lowered FACE {face_id} with {stored_face['confidence']} and {stored_face['stability']}")
                    # print(", ".join(class_names.get(sticker['class_id'], "Unknown") for sticker in ring))
                if stored_face['confidence'] < 0.1:
                    stored_face['ring'] = ring
                    stored_face['center'] = center
                    stored_face['confidence'] = 0.3
                    stored_face['stability'] = 0
                    # print(f"confidence reset FACE {face_id}")
                    # print(", ".join(class_names.get(sticker['class_id'], "Unknown") for sticker in ring))
        return trust

    def update_face(self, grouped_faces, frame):
        face1, face2 = grouped_faces[0], grouped_faces[1]
        center1 = face1[0]
        center2 = face2[0]
        face_id1 = classify_center_color(frame, center1['bbox'])
        face_id2 = classify_center_color(frame, center2['bbox'])
        if (face_id1 not in FACE_NEIGHBORS[face_id2]):
            print("Two faces are the same, skip")
            return
        perimeter1 = face1[1:]
        perimeter2 = face2[1:]
        conf1 = self.check_face(face_id1, center1, perimeter1)
        conf2 = self.check_face(face_id2, center2, perimeter2)
        
        if conf1 and conf2:
            self.unify_edge(face_id1, face_id2, center1, center2, perimeter1, perimeter2)

    def unify_edge(self, c1, c2, center1, center2, ring1, ring2):
        if (self.faces[c1]['confidence'] < 0.8) or (self.faces[c2]['confidence'] < 0.8) or (self.faces[c1]['stability'] < 5) or (self.faces[c2]['stability'] < 5):
            print("No enough confidence or stability to unify edge")
            return

        dx = center2['cx'] - center1['cx']
        dy = center2['cy'] - center1['cy']
        magnitude = np.hypot(dx, dy)
        if magnitude == 0:
            directionc1c2 = (0, 0)
        else:
            directionc1c2 = (dx / magnitude, dy / magnitude)
        directionc2c1 = (-directionc1c2[0], -directionc1c2[1])

        best_score = -float('inf')
        best_triple_idx = None 
        best_triple = None
        for i in range(8):
            i2 = (i + 1) % 8
            i3 = (i + 2) % 8
            triplet = ring1[i], ring1[i2], ring1[i3]
            align_score = self.score_triplet(triplet, directionc1c2, center1)
            dis_score = -self.average_distance(triplet, center2)
            score = align_score*0.4 + dis_score*0.6
            print(f"Triplet {triplet} of face {c1} to face {c2}- Align Score: {align_score:.3f} - dis Score: {dis_score:.3f} and final score: {score:.3f}")
            if score > best_score:
                best_score = score
                best_triple_idx = (i, i2, i3)
                best_triple = triplet
        print(f"Best triplet for face {c1} to face {c2} is {best_triple} with score {best_score:.3f}")
        
        
        # do the opposite on c2
        best_dist2 = -float('inf')
        best_triple_idx2 = None
        best_triple2 = None
        for i in range(8):
            i2 = (i + 1) % 8
            i3 = (i + 2) % 8
            triplet = ring2[i], ring2[i2], ring2[i3]
            align_score = self.score_triplet(triplet, directionc2c1, center2)
            dis_score = -self.average_distance(triplet, center1)
            score = align_score*0.4 + dis_score*0.6
            print(f"Triplet {triplet} of face {c2} to face {c1}- Align Score: {align_score:.3f} - dis Score: {dis_score:.3f} and final score: {score:.3f}")
            if score > best_dist2:
                best_dist2 = score
                best_triple_idx2 = (i, i2, i3)
                best_triple2 = triplet
        print(f"Best triplet for face {c2} to face {c1} is {best_triple2} with score {best_dist2:.3f}")

        # if face ring is finalized, find the correspond indexing of the triple
        print(f"Before checking ring_finalized for {c1} and {c2}")
        print(f"Ring for {c1}: {[class_names.get(sticker['class_id'], 'Unknown') for sticker in self.faces[c1]['ring']]}")
        print(f"Ring for {c2}: {[class_names.get(sticker['class_id'], 'Unknown') for sticker in self.faces[c2]['ring']]}")

        if self.faces[c1]['ring_finalized'] == True:
            idx = self.get_ring_idx(best_triple, self.faces[c1]['ring'])
            print(f"Ring index for best_triple in {c1}: {idx}")
            if idx is None:
                return
            else:
                best_triple_idx = idx
            
        if self.faces[c2]['ring_finalized'] == True:
            idx = self.get_ring_idx(best_triple2, self.faces[c2]['ring'])
            print(f"Ring index for best_triple2 in {c2}: {idx}")
            if idx is None:
                return
            else:
                best_triple_idx2 = idx
        print(f"After checking ring_finalized for {c1} and {c2}")
        print(f"Best triple index for {c1}: {best_triple_idx}")
        print(f"Best triple index for {c2}: {best_triple_idx2}")

        self.faces[c1]['ring_finalized'] = True
        self.faces[c2]['ring_finalized'] = True

        prev_edge1 = self.faces[c1]['neighbors'][c2]
        if prev_edge1:
            prev_triple = prev_edge1['my_triple']
            if prev_triple == best_triple or prev_triple == [best_triple[2], best_triple[1], best_triple[0]]:
                self.faces[c1]['neighbors'][c2]['confidence'] = min(1.0, prev_edge1['confidence'] + 0.1)
            else:
                self.faces[c1]['neighbors'][c2]['confidence'] = min(1.0, prev_edge1['confidence'] - 0.1)
            if self.faces[c1]['neighbors'][c2]['confidence'] < 0.2:
                print(f"Face {c1} with neighbor {c2} confidence reset")
                self.faces[c1]['neighbors'][c2] = {
                'my_triple_idx': best_triple_idx,
                'my_triple': best_triple,
                'their_triple_idx': best_triple_idx2,
                'their_triple': best_triple2,
                'confidence': 0.3
            }
        else: # new edge, initialize confidence
            print(f"[DEBUG] Face {c1} added edges:{[class_names.get(sticker['class_id'], 'Unknown') for sticker in best_triple]} with face {c2}")
            self.faces[c1]['neighbors'][c2] = {
                'my_triple_idx': best_triple_idx,
                'my_triple': best_triple,
                'their_triple_idx': best_triple_idx2,
                'their_triple': best_triple2,
                'confidence': 0.3
            }
            
        prev_edge2 = self.faces[c2]['neighbors'][c1]
        if prev_edge2:
            prev_triple = prev_edge2['my_triple']
            if prev_triple == best_triple2 or prev_triple == [best_triple2[2], best_triple2[1], best_triple2[0]]:
                self.faces[c2]['neighbors'][c1]['confidence'] = min(1.0, prev_edge2['confidence'] + 0.1)
            else:
                self.faces[c2]['neighbors'][c1]['confidence'] = min(1.0, prev_edge2['confidence'] - 0.1)
            # if self.faces[c2]['neighbors'][c1]['confidence'] < 0.2:
            #     print(f"Face {c2} with neighbor {c1} confidence reset")
            #     self.faces[c2]['neighbors'][c1] = {
            #     'my_triple_idx': best_triple_idx2,
            #     'my_triple': best_triple2,
            #     'their_triple_idx': best_triple_idx,
            #     'their_triple': best_triple,
            #     'confidence': 0.3
            # }
        else: # new edge, initialize confidence
            print(f"[DEBUG] Face {c2} added edges:{[class_names.get(sticker['class_id'], 'Unknown') for sticker in best_triple2]} with face {c1}")
            self.faces[c2]['neighbors'][c1] = {
                'my_triple_idx': best_triple_idx2,
                'my_triple': best_triple2,
                'their_triple_idx': best_triple_idx,
                'their_triple': best_triple,
                'confidence': 0.3
            }


        self.check_finalize(c1)
        self.check_finalize(c2)

    def check_finalize(self, face):
        if face not in self.faces or self.faces[face]['finalized'] == True:
            print(f"quit at check finalize for face {face}")
            return
        
        valid_neighbors = [n for n, data in self.faces[face]['neighbors'].items() if data is not None]
        adjacent_pairs = []
        for i in range(len(valid_neighbors)):
            for j in range(i+1, len(valid_neighbors)):
                nei1, nei2 = valid_neighbors[i], valid_neighbors[j]
                if valid_neighbors[j] in FACE_NEIGHBORS[valid_neighbors[i]]: #and self.faces[face]['neighbors'][nei1]['confidence'] > 0.7 and self.faces[face]['neighbors'][nei2]['confidence'] > 0.7:
                    adjacent_pairs = (valid_neighbors[i], valid_neighbors[j])
                    break
        if adjacent_pairs:
            print(f"Get Finalized face {face}")
            self.finalize(face, adjacent_pairs)
    
    def finalize(self, face, pair):
        # finalize the 3x3 matrix 
        nei1, nei2 = pair
        
        final_matrix = self.construct_matrix(face, nei1, nei2)
        if final_matrix is not None:
            print(f"For Face {face}, matrix is {final_matrix}")
            self.faces[face]['finalized'] == True
            self.net[face] = final_matrix


    def construct_matrix(self, face, nei1, nei2):
        # Initialize an empty 3x3 grid
        matrix = [[None, None, None],
                [None, face, None],
                [None, None, None]]
        key = tuple(sorted([nei1, nei2]))
        corner_idx = corner_mapping[face][key]
        print(f'corner idx is {corner_idx} of face {face}')
        if corner_idx in {0, 2}:
            top = FACE_NEIGHBORS[face][1]
            edge_indices = self.faces[face]['neighbors'][top]['my_triple']
        else:
            bottom = FACE_NEIGHBORS[face][3]
            edge_indices = self.faces[face]['neighbors'][bottom]['my_triple']
        print(f"[DEBUG] Edge indices for {face}: {[class_names.get(sticker['class_id'], 'Unknown') for sticker in edge_indices]}")
        edge_indiced_idx1 = self.faces[face]['neighbors'][nei1]['my_triple_idx']
        edge_indiced_idx2 = self.faces[face]['neighbors'][nei2]['my_triple_idx']
        corner = None
        print(f"index of {nei1} is {edge_indiced_idx1}")
        print(f"index of {nei2} is {edge_indiced_idx2}")
        if edge_indiced_idx1[2] == edge_indiced_idx2[0]:
            corner = edge_indiced_idx1[2]
        elif edge_indiced_idx1[0] == edge_indiced_idx2[2]:
            corner = edge_indiced_idx1[0]
        if corner is None:
            print("ERROR. Detection Wrong, no corner common sticker found")
            print(f"ring of face {face} is  + {self.faces[face]['ring']}")
            return
        print(f'[DEBUG] edge idx is {corner}')
        # print(f"ring is  + {self.faces[face]['ring']}")
        # self.rotate_ring(face, corner)
        stickers = self.faces[face]['ring']
        
        
        # print("Edge is:", ", ".join(class_names.get(sticker['class_id'], "Unknown") for sticker in edge_indices))
        # print("ring is " + ", ".join(class_names.get(sticker['class_id'], "Unknown") for sticker in stickers))
        print(f"ring of face {face} is  + {stickers}")
        



        # Determine the row/column orientation based on the edge and corner position


        if corner_idx in {0, 2}:  # Top-left or Top-right
            if corner_idx == 0:
                # Place stickers clockwise
                matrix[0][0] = class_names.get(stickers[0]['class_id'])
                matrix[0][1] = class_names.get(stickers[1]['class_id'])
                matrix[0][2] = class_names.get(stickers[2]['class_id'])
                matrix[1][0] = class_names.get(stickers[7]['class_id']) # Left col
                matrix[2][0] = class_names.get(stickers[6]['class_id'])
                matrix[1][2] = class_names.get(stickers[3]['class_id'])  # Right col
                matrix[2][2] = class_names.get(stickers[4]['class_id'])
                matrix[2][1] = class_names.get(stickers[5]['class_id'])  # Bottom center
            else:
                # Place stickers counterclockwise
                matrix[0][2] = class_names.get(stickers[0]['class_id'])
                matrix[0][1] = class_names.get(stickers[1]['class_id'])
                matrix[0][0] = class_names.get(stickers[2]['class_id'])
                matrix[1][2] = class_names.get(stickers[7]['class_id']) # Left col
                matrix[2][2] = class_names.get(stickers[6]['class_id'])
                matrix[1][0] = class_names.get(stickers[3]['class_id'])  # Right col
                matrix[2][0] = class_names.get(stickers[4]['class_id'])
                matrix[2][1] = class_names.get(stickers[5]['class_id'])  # Bottom center

        elif corner_idx in {6, 8}:  # Bottom-left or Bottom-right
            if corner_idx == 6:
                # Place stickers clockwise
                matrix[2][0] = class_names.get(stickers[0]['class_id'])
                matrix[2][1] = class_names.get(stickers[1]['class_id'])
                matrix[2][2] = class_names.get(stickers[2]['class_id'])
                matrix[0][0] = class_names.get(stickers[6]['class_id']) # Left col
                matrix[1][0] = class_names.get(stickers[7]['class_id'])
                matrix[0][2] = class_names.get(stickers[4]['class_id'])  # Right col
                matrix[1][2] = class_names.get(stickers[3]['class_id'])
                matrix[0][1] = class_names.get(stickers[5]['class_id'])  # Bottom center
            else:
                # Place stickers counterclockwise
                matrix[2][2] = class_names.get(stickers[0]['class_id'])
                matrix[2][1] = class_names.get(stickers[1]['class_id'])
                matrix[2][0] = class_names.get(stickers[2]['class_id'])
                matrix[0][2] = class_names.get(stickers[6]['class_id']) # Left col
                matrix[1][2] = class_names.get(stickers[7]['class_id'])
                matrix[0][0] = class_names.get(stickers[4]['class_id'])  # Right col
                matrix[1][0] = class_names.get(stickers[3]['class_id'])
                matrix[0][1] = class_names.get(stickers[5]['class_id'])  # Bottom center

        return matrix

    def score_triplet(self, triplet, direction, center):
        cx = np.mean([st['cx'] for st in triplet])
        cy = np.mean([st['cy'] for st in triplet])
        
        # Vector from face center to triplet centroid
        # centroid_vector = (cx - triplet[0]['cx'], cy - triplet[0]['cy'])
        # magnitude = np.hypot(centroid_vector[0], centroid_vector[1])

        # if magnitude == 0:
        #     print("Warning: Magnitude is zero, returning 0 for alignment score.")
        #     return 0  # Avoid division by zero

        # # Normalize the vector
        # centroid_vector = (centroid_vector[0] / magnitude, centroid_vector[1] / magnitude)
        # # Compute dot product to measure alignment with expected direction
        # dot_product = centroid_vector[0] * direction[0] + centroid_vector[1] * direction[1]
        
        second_sticker = triplet[1]
        edge_vector = np.array([second_sticker['cx'] - center['cx'], second_sticker['cy'] - center['cy']])
        edge_magnitude = np.linalg.norm(edge_vector)
        if edge_magnitude == 0:
            print("Warning: Edge magnitude is zero, returning 0 for alignment score.")
            return 0
        edge_vector = edge_vector / edge_magnitude  # Normalize the edge vector
        align_score = np.dot(edge_vector, direction)
        # angle = np.arccos(dot_product)
        # align_score = np.abs(np.sin(angle))
        return align_score
    
    def average_distance(self, triplet, center):
        total_dist = 0
        for sticker in triplet:
            dx = sticker['cx'] - center['cx']
            dy = sticker['cy'] - center['cy']
            total_dist += np.hypot(dx, dy)
        ave_dist = total_dist
        normalize_dist = (ave_dist - 10) / (900 - 10)
        return 1 - normalize_dist
    
    def get_ring_idx(self, triple, ring):
        triple_colors = [class_names[sticker['class_id']] for sticker in triple]
        ring_colors = [class_names[sticker['class_id']] for sticker in ring]
        print(f"Finding indices for triplet: {triple_colors}")
        print(f"Ring: {ring_colors}")
        for i in range(8):
            i2, i3 = (i + 1) % 8, (i + 2) % 8
            if [ring_colors[i], ring_colors[i2], ring_colors[i3]] == triple_colors:
                print(f"Match found at indices: {(i, i2, i3)}")
                return (i, i2, i3)
            if [ring_colors[i], ring_colors[i2], ring_colors[i3]] == triple_colors[::-1]:
                print(f"Match found at indices: {(i3, i2, i)}")
                return (i3, i2, i)
        
        print("No match found")
        return None
    
    def rotate_ring(self, face_id, new_index):
        ring = self.faces[face_id]['ring']
        ring = ring[new_index:] + ring[:new_index]
        self.faces[face_id]['ring'] = ring


def compute_face_similarity(ring1, ring2):
                best = 0.0
                for offset in range(8):  # Try all possible rotations
                    rotated_ring2 = ring2[offset:] + ring2[:offset]
                    matches = sum(1 for s1, s2 in zip(ring1, rotated_ring2) 
                                if s1['class_id'] == s2['class_id'])
                    best = max(best, matches / len(ring1))
                return best
                
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
        ((10, 120, 50), (23, 255, 255))
    ],
    "green": [
        ((35, 50, 50), (90, 255, 255))
    ],
    "blue": [
        ((90, 100, 50), (140, 255, 255))      # Only 1 subrange, but in a list
    ],
    "yellow": [
        ((23, 150, 150), (40, 255, 255))
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

def update_ring(center, perimeter):

        cx_center = center['cx']
        cy_center = center['cy']

        peri = []
        for st in perimeter:
            dx = st['cx'] - cx_center
            dy = st['cy'] - cy_center
            angle = math.atan2(dy, dx)
            peri.append((st, angle))
        peri.sort(key = lambda x: x[1]) # sort by angle ascending
        return [item[0] for item in peri]

def main():
    yolo_model = YOLO("cube-vision/runs/detect/train19/weights/best.pt")
    face_data = {
        "white": None,
        "yellow": None,
        "blue": None,
        "green": None,
        "red": None,
        "orange": None
    }
    cube_state = CubeState()
    cap = cv2.VideoCapture(0)  # Use 0 if DroidCam is set as the default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: could not read frame")
            break
        
        # YOLO detection
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

        # group faces based on center sticker
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

            nearest_8 = [info[1] for info in distances[:8]] if len(distances)>=8 else [info[1] for info in distances]
            if len(nearest_8) < 8:
                continue
            if filter_center(center_det, nearest_8):
                faceA = [center_det] + nearest_8
                grouped_faces.append(faceA)
            face_id = classify_center_color(frame, center_det['bbox'])
            cube_state.check_face(face_id, center_det, nearest_8)

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
            if len(nearest_8A) < 8:
                continue
            if filter_center(centerA, nearest_8A):
                faceA = [centerA] + nearest_8A
                grouped_faces.append(faceA)
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
            if len(nearest_8B) < 8:
                continue
            if filter_center(centerB, nearest_8B):
                faceB = [centerB] + nearest_8B
                grouped_faces.append(faceB)

            if len(grouped_faces) != 2:
                continue
            cx1, cy1 = centerA['cx'], centerA['cy']
            cx2, cy2 = centerB['cx'], centerB['cy']

            dist = np.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)

            # Dynamic threshold: Use avg spacing of nearby stickers to determine a valid min distance
            avg_sticker_size = np.mean([centerA['bbox'][2] - centerA['bbox'][0], 
                                        centerB['bbox'][2] - centerB['bbox'][0]])  # Avg width
            min_dist = avg_sticker_size * 1.5  # Ensure at least this much spacing between two center stickers
            
            if not dist >= min_dist:
                continue

            nearest_8A_new = check_overlap(nearest_8A, nearest_8B)
            if len(nearest_8A) != len(nearest_8A_new): # there is no overlapping detection between two faces
                continue
            cube_state.update_face(grouped_faces, frame)

        else:  # more than two faces, somethings thing is wrong, ignore this frame
            pass    
        


        # draw bbox of each detected face
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
                    cls_id = classify_center_color(frame, st['bbox'])
                else:
                    # normal sticker => face color
                    rect_color = color
                    thickness = 2

                # draw
                cv2.rectangle(frame, (x1,y1), (x2,y2), rect_color, thickness)
                label_str = f"cls={cls_id}"
                cv2.putText(frame, label_str, (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, rect_color, 2)

        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print("------Final cube state------")
    for face, data in cube_state.faces.items():
            print(f"Face {face}: Confidence: {data['confidence']:.2f}, Stability: {data['stability']}, Ring Finalized: {data['ring_finalized']}")
            ring_colors = ", ".join(class_names.get(sticker['class_id'], "Unknown") for sticker in data['ring'])
            print(f"Ring: {ring_colors}")
            for neighbor, edge in data['neighbors'].items():
                if edge is not None:
                    print(f" - Edge between {face} and {neighbor}:")
                    print(", ".join(class_names.get(sticker['class_id'], "Unknown") for sticker in edge['my_triple']))
            print() 
    print(f"Final net is: {cube_state.net}")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()