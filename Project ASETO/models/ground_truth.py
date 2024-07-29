import os
import csv
import glob

# Path to annotation files and output CSV
annotations_folder = r'D:\Pengujian_Tracking\test'
output_csv = r'D:\Pengujian_Tracking\ground_truths_update.csv'
img_width = 640  # Replace with the actual width of the images
img_height = 480  # Replace with the actual height of the images

# Ensure the output directory exists
output_dir = os.path.dirname(output_csv)
os.makedirs(output_dir, exist_ok=True)

# Function to convert YOLO format to bounding box coordinates (x1, y1, x2, y2)
def yolo_to_bbox(yolo_bbox, img_width, img_height):
    class_id, x_center, y_center, width, height = yolo_bbox
    x1 = int((x_center - width / 2) * img_width)
    y1 = int((y_center - height / 2) * img_height)
    x2 = int((x_center + width / 2) * img_width)
    y2 = int((y_center + height / 2) * img_height)
    return (class_id, x1, y1, x2, y2)

# Function to load ground truths from annotation files
def load_ground_truths(annotation_folder):
    ground_truths = {}
    for file_path in glob.glob(os.path.join(annotation_folder, '*.txt')):
        filename = os.path.basename(file_path).split('.')[0]
        try:
            frame_number = int(filename)  # Filename should be frame number, e.g., 0.txt, 1.txt
        except ValueError:
            print(f"Skipping file with invalid frame number: {filename}")
            continue
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"Skipping invalid line in {file_path}: {line}")
                    continue
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                bbox = (class_id, x_center, y_center, width, height)
                converted_bbox = yolo_to_bbox(bbox, img_width, img_height)
                if frame_number not in ground_truths:
                    ground_truths[frame_number] = []
                ground_truths[frame_number].append(converted_bbox)
    return ground_truths

# Load ground truths from annotation folder
ground_truths = load_ground_truths(annotations_folder)

# Convert ground truths to list format
ground_truth_list = [(frame_number, bboxes) for frame_number, bboxes in sorted(ground_truths.items())]

# Print ground truths in the desired format
for entry in ground_truth_list:
    frame_number, bboxes = entry
    bbox_list = [(class_id, x1, y1, x2, y2) for (class_id, x1, y1, x2, y2) in bboxes]
    print(f"({frame_number}, {bbox_list})")

# Optionally save to CSV file in a readable format
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['frame_number', 'bounding_boxes'])
    for frame_number, bboxes in ground_truth_list:
        bbox_str = '; '.join([f"({class_id}, {x1}, {y1}, {x2}, {y2})" for class_id, x1, y1, x2, y2 in bboxes])
        writer.writerow([frame_number, bbox_str])

print(f"Ground truths have been saved to {output_csv}")
