import cv2
import numpy as np
import time
import csv
import os
from ultralytics import YOLO

# Path to the folder containing frames
frames_folder = 'D:/Pengujian_Tracking/Frame Rate/FramesRate15/'
csv_filename = 'detailed_metrics_per_frame.csv'

# Load YOLOv8 model (ensure you have the model weights)
model = YOLO('../YOLO-Weights/detection_model_optimization.pt')

# Initialize metrics
frame_count = 0
start_time = time.time()
processing_times = []
id_switches_total = 0
false_positives_total = 0
false_negatives_total = 0
true_positives_total = 0
iou_sum_total = 0
total_ground_truth_objects = 0
previous_tracks = {}

# Ground truths for evaluation (this should be provided)
ground_truths = {
    # Format: {frame_number: [bounding_boxes]}
    0: [(52,57,592,311),(383,317,546,462),(88,313,342,466),(56,42,323,412),(15,102,623,385), (35,94,605,480), (25,47,620,429),
        (32,164,598,406), (16,153,580,428), (18,64,614,327), (61,52,464,453), (7,153,633,480), (60,122,584,407), (15,9,532,458),
        (12,73,622,381), (56,11,533,473), (48,32,547,455), (4,7,640,446)],
    1: [(15, 216, 319, 479), (114,72,571,420), (14,41,620,468), (61,45,603,446),(28,27,632,479),(185,90,639,480),
         (25,2,609,454), (43,4,632,440), (28,27,632,479),(21,10,623,448), (17,0,640,459), (26,92,621,382), (12,55,555,478),
         (82,52,590,405),(51,76,638,480), (100,8,549,455), (0,8,640,480), (63,4,595,457)],
    2: [(25,84,622,407), (75,8,563,459), (77,18,608,462), (25,13,606,472), (65,49,624,477),
         (36,18,596,456),(6,16,611,458), (14,30,611,461), (50,52,617,459), (3,0	,615,471), (53,72,602,393),
         (48,4,556,474), (29,9,623,460), (19,53,627,446), (114,92,525,399), (91,5,559,472),
         (25,19,607,435), (31,17,613,459)],
    3: [(166,1,444,459), (15,101,599,402), (154,93,601,480), (192,9,449,480), (144,5,504,480),
         (202,0,475,480), (0,82,564,463), (45,0,640,480), (174,0,490,480), (0,23,640,422),
         (0,59,640,375),(174,3,440,462), (154,76,514,456), (170,60,508,461), (75,7,587,469), (66,10,541,471),
         (10,111,615,385), (9,58,634,361)],
    4: [(139,78,507,401), (89,112,592,378), (136,38,521,435),
         (367,55,495,292), (231,133,377,331),(498,131,640,372), (275,337,608,421), (505,26,632,330),
         (96,3,535,479), (160,37,361,324), (20,2,634,453), (111,210,420,465), (156,36,468,449),
         (9,26,526,480), (207,109,457,399), (174,41,617,463),(183,23,451,466), (212,61,420,468)],
    5: [(146,30,486,464),(157, 46, 407, 472),(197, 87, 460, 476),(143,58,482,423), (199,15,459,479), (5,14,602,464), (146,5,516,480), (5,66,640,419),
         (323,7,630,480), (150,2,494,480), (207,27,434,441), (26,5,624,474), (320,9,520,480),
         (184,33,451,437), (144,2,485,479), (200,60, 550,395), (135,45,510,370), (157,37,515,418),
         (83,66,553,415), (323,7,630,480), (150,2,494,480), (207,27,434,441), (26,5,624,474), (320,9,520,480),
         (184,33,451,437), (200,60,550,395), (135,45,510,370), (157,37,515,418), (83,66,553,415)],
    6: [(36,108,592,341), (151,81,564,464), (367,192,640,480), (41,64,608,437), (160,40,574,459),
         (39,33,640,452),(38,54,574,465), (146,153,560,428), (67,63,492,381), (42,88,640,464),
         (320, 9, 520, 480),(93,149,508,463), (41,42,591,461), (49,71,605,401), (19,56,616,441), (21,192,638,480),
         (21,88,605,479), (42,124,591,450)],
    7: [(66,215,436,435), (13,39,530,436), (74,23,612,480), (66,171,406,406), (7,51,615,441),
         (55,44,624,410), (45,192,626,313), (36,331,639,471), (104,13,209,397), (228,4,319,480),
         (324,1,424,478), (409,12,513,472), (64,20,640,457), (60,79,545,342), (70,156,529,340),
         (156,22,458,431), (21,21,621,462), (26,53,630,413), (15,9,70,458), (82,10,134,447),
         (123,13,193,459), (193,9,268,453), (257,11,337,457), (324,6,389,449), (386,11,438,454),
         (449,9,503,461), (517,15,564,452), (571,5,632,454), (17,8,607,439), (99,4,474,393),
         (65,74,640,335), (11,86,626,309), (225,47,420,437), (87,42,466,476), (2, 251, 639, 479), (23, 302, 639, 453),
         (10, 296, 638, 465), (6, 296, 633, 453), (18, 295, 618, 459), (17, 296, 609, 457), (27, 299, 599, 460),
         (20, 294, 603, 456), (19, 300, 609, 465),(28, 305, 613, 466), (5, 303, 583, 451), (15, 305, 619, 453), (14, 308, 604, 455)
         ,(14, 303, 559, 458), (10, 297, 575, 447), (13, 297, 592, 464), (8, 300, 615, 466), (8, 296, 633, 471), (27, 295, 630, 469),
         (14, 294, 619, 473), (12, 288, 603, 467), (14, 287, 623, 465), (16, 293, 590, 456), (15, 297, 582, 466), (8, 292, 597, 461),
         (11, 297, 574, 457), (13, 298, 604, 452), (14, 299, 596, 452),(14, 306, 613, 458), (18, 318, 629, 453), (19, 297, 627, 420),
         (14, 280, 621, 433), (11, 270, 630, 450), (14, 277, 617, 436), (11, 254, 626, 454), (12, 256, 623, 449), (15, 268, 639, 468),
         (17, 279, 638, 470), (15, 283, 628, 460), (18, 284, 591, 461), (17, 287, 616, 454), (11, 289, 559, 447), (11, 300, 587, 456),
         (15, 300, 600, 461), (15, 296, 568, 447), (13, 287, 630, 465), (7, 270, 585, 450), (10, 284, 597, 438), (16, 209, 554, 415)],
    8: [(159,118,491,400), (22,24,625,465), (111,79,534,389), (64,38,574,441), (30,46,603,434),
         (68,4,612,470), (54,35,532,446), (112,27,578,392), (104, 13, 209, 397), (228, 4, 319, 480),
         (324, 1, 424, 478), (409, 12, 513, 472), (64, 20, 640, 457), (60, 79, 545, 342), (70, 156, 529, 340),
         (156, 22, 458, 431), (21, 21, 621, 462), (26, 53, 630, 413), (15, 9, 70, 458), (82, 10, 134, 447),
         (123, 13, 193, 459), (193, 9, 268, 453), (257, 11, 337, 457), (324, 6, 389, 449), (386, 11, 438, 454),
         (449, 9, 503, 461), (517, 15, 564, 452), (571, 5, 632, 454), (17, 8, 607, 439), (99, 4, 474, 393),
         (65, 74, 640, 335), (11, 86, 626, 309), (225, 47, 420, 437), (87, 42, 466, 476)],
    9: [(64,76,603,411), (23,56,628,418), (25,42,595,388), (170,10,454,457), (22,34,626,466),
         (115,101,607,385), (167,226,512,407), (214,110,603,293), (154,205,516,370), (21,42,619,413),
         (154,198,403,405), (193,196,463,343), (26,55,631,422), (156,66,493,433), (56,138,560,393),
         (137,199,498,367), (162,11,487,451), (124,17,47,431), (34,33,569,440), (77,20,609,452),
         (57,32,592,449)],
    10: [(12,30,640,469), (20,34,640,445), (108,65,557,438), (43,16,606,446), (24,18,622,388),
         (13,44,606,428), (40,32,592,407), (31,100,542,456), (124,114,520,434), (14,17,575,453)],
    11: [(99,59,545,430), (21, 68, 562, 477),(68,256,359,448), (20,14,625,469), (56,122,560,414), (17,26,603,453),
          (159,42,534,442),(20, 74, 562, 476),(18, 77, 557, 474),(20, 62, 567, 477), (17, 58, 571, 477),(10, 70, 564, 478),
          (16, 108, 538, 474),(15, 52, 568, 474),(13, 55, 568, 477),(42,16,587,432), (315,11,534,470), (35,5,531,468), (12,38,633,464),
          (151,103,471,351),(23, 63, 564, 477),(16, 44, 576, 475),(19, 59, 562, 478),(21, 47, 579, 476),(22, 56, 561, 478),(20, 40, 576, 476),
          (19, 61, 560, 477), (23, 75, 565, 478),(24, 50, 568, 476),(15, 44, 569, 474),(11, 15, 569, 475),(12, 44, 561, 477),(30,16,609,451), (350,198,618,425),
          (88,7,551,461), (32,8,616,467),(64,38,590,432), (23, 98, 559, 478),(19, 72, 561, 476),(23, 61, 562, 478),(16, 66, 562, 477),
          (27,84,607,386), (51,18,608,321), (18, 42, 566, 477), (14, 45, 579, 475), ],
    12: [(29,13,640,433), (19,14,607,444), (35,23,618,473), (105,69,537,374), (63,172,610,375),
          (7,20,640,473), (2,11,639,476), (25,127,526,469), (56,150,586,295), (74,54,518,411),
          (11,11,611,475), (65,150,579,371), (109,65,515,453), (109,20,520,457), (32,8,616,467),
          (62,3,533,480)],
    13: [(204,51,407,339), (253,4,386,480), (53,36,195,406), (266,39,399,411), (469,48,598,414),
          (41,151,625,335), (179,5,482,480), (206,2,443,449), (78,24,269,417), (234,52,393,429),
          (238,3,431,475), (252,1,392,467), (63,122,598,350), (155,32,466,446), (83,166,559,336),
          (286,27,555,480)],
    14: [(18,8,615,463), (17,11,634,460), (32,23,620,461), (9,34,640,460), (29,11,598,469),
          (51,72,629,419), (27,111,486,456), (163,6,493,464), (19,30,533,404), (22,7,640,479),
          (25,106,542,392), (8,15,608,456), (30,19,606,479), (321,267,489,390),
          (83, 166, 559, 336),(327,46,435,187), (36,62,316,174),(442,67,614,171),(7,271,337,387),
          (22,8,63,468), (49,126,579,365), (38,24,569,449), (7,52,640,438)],
    15: [(108,34,585,458), (75,106,323,390), (13,14,628,453), (5,16,640,455), (30,4,640,477),
          (56,36,598,423), (145,9,556,373), (14,18,640,456), (66,16,604,435), (12,2,640,480),
          (57,56,598,408), (60,34,604,439), (24,6,628,475), (12,5,630,470),
          (71,8,629,478), (44,12,633,377), (33,16,548,460), (68,24,602,464), (2,74,639,392)],
}

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_gt, y1_gt, x2_gt, y2_gt = box2

    xi1 = max(x1, x1_gt)
    yi1 = max(y1, y1_gt)
    xi2 = min(x2, x2_gt)
    yi2 = min(y2, y2_gt)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def calculate_metrics(detections, ground_truths, iou_threshold=0.5):
    id_switches = 0  # ID switches belum dihitung dalam contoh ini
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    iou_sum = 0
    ious_per_true_positive = []

    matched_gt = set()
    for det in detections:
        best_iou = 0
        best_gt_idx = -1
        for idx, gt in enumerate(ground_truths):
            if idx in matched_gt:
                continue
            iou = calculate_iou(det, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_iou >= iou_threshold:
            true_positives += 1
            matched_gt.add(best_gt_idx)
            iou_sum += best_iou
            ious_per_true_positive.append(best_iou)
        else:
            false_positives += 1

    false_negatives = len(ground_truths) - len(matched_gt)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    mota = 1 - (false_negatives + id_switches + false_positives) / len(ground_truths) if len(ground_truths) > 0 else 0
    motp = iou_sum / true_positives if true_positives > 0 else 0

    return id_switches, false_positives, false_negatives, mota, precision, recall, iou_sum, true_positives, motp, ious_per_true_positive

def calculate_id_switches(current_tracks, previous_tracks):
    id_switches = 0
    for track_id, bbox in current_tracks.items():
        if track_id in previous_tracks:
            if previous_tracks[track_id] != bbox:
                id_switches += 1
    return id_switches

# Create a CSV file to save detailed metrics per frame
with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ['Frame Number', 'Processing Time (s)', 'False Positives', 'False Negatives', 'True Positives', 'ID Switches', 'IoU Sum', 'MOTA', 'Precision', 'Recall', 'MOTP', 'IoU per True Positive']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Read frames from the frames folder
    for frame_filename in sorted(os.listdir(frames_folder)):
        if not frame_filename.endswith('.jpg'):
            continue

        frame_path = os.path.join(frames_folder, frame_filename)
        img = cv2.imread(frame_path)

        frame_start_time = time.time()
        frame_number = int(frame_filename.split('_')[1].split('.')[0])

        # Detect objects in the frame using YOLOv8
        results = model(img)

        # Extract detections from results
        detections = results[0].boxes.xyxy.cpu().numpy()  # Assuming results is a list and boxes is an attribute
        detections = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in detections]

        # Ground truths for current frame
        ground_truths_current_frame = ground_truths.get(frame_number, [])

        # Calculate metrics
        id_switches, false_positives, false_negatives, mota, precision, recall, iou_sum, true_positives, motp, ious_per_true_positive = calculate_metrics(detections, ground_truths_current_frame)

        # Update totals for averaging later
        id_switches_total += id_switches
        false_positives_total += false_positives
        false_negatives_total += false_negatives
        total_ground_truth_objects += len(ground_truths_current_frame)
        iou_sum_total += iou_sum
        true_positives_total += true_positives

        # Calculate processing time per frame
        processing_time = time.time() - frame_start_time
        processing_times.append(processing_time)

        # Increase frame counter
        frame_count += 1

        # Print debug info for each frame
        print(f"Frame {frame_count}: processing_time={processing_time:.4f} seconds, detections={detections}")

        # Write detailed metrics for the current frame to CSV
        writer.writerow({
            'Frame Number': frame_number,
            'Processing Time (s)': processing_time,
            'False Positives': false_positives,
            'False Negatives': false_negatives,
            'True Positives': true_positives,
            'ID Switches': id_switches,
            'IoU Sum': iou_sum,
            'MOTA': mota,
            'Precision': precision,
            'Recall': recall,
            'MOTP': motp,
            'IoU per True Positive': ious_per_true_positive
        })

# Calculate and print overall metrics
elapsed_time = time.time() - start_time
average_frame_rate = frame_count / elapsed_time if elapsed_time > 0 else 0
average_processing_time = sum(processing_times) / len(processing_times) if len(processing_times) > 0 else 0
mota_total = 1 - (false_negatives_total + id_switches_total + false_positives_total) / total_ground_truth_objects if total_ground_truth_objects > 0 else 0
precision_total = true_positives_total / (true_positives_total + false_positives_total) if (true_positives_total + false_positives_total) > 0 else 0
recall_total = true_positives_total / (true_positives_total + false_negatives_total) if (true_positives_total + false_negatives_total) > 0 else 0
motp_total = iou_sum_total / true_positives_total if true_positives_total > 0 else 0

print(f"Elapsed time: {elapsed_time:.2f} seconds")
print(f"Average frame rate: {average_frame_rate:.2f} frames per second")
print(f"Average processing time per frame: {average_processing_time:.4f} seconds")
print(f"Total ID switches: {id_switches_total}")
print(f"Total false positives: {false_positives_total}")
print(f"Total false negatives: {false_negatives_total}")
print(f"Total IoU sum: {iou_sum_total}")
print(f"Total true positives: {true_positives_total}")
print(f"MOTA: {mota_total:.4f}")
print(f"Precision: {precision_total:.4f}")
print(f"Recall: {recall_total:.4f}")
print(f"MOTP: {motp_total:.4f}")
