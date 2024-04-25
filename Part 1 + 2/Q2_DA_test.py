import os
import shutil
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

class BoundingBox:
    def __init__(self, x_center=0, y_center=0, width=0, height=0, iou=0, label='None'):
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height
        self.iou = iou
        self.label = label


def read_lean_map_of_bboxes(input_json_file_path: str):
    with open(input_json_file_path, 'r') as json_stream:
        raw_object = json.load(json_stream)
    return {k: [BoundingBox(**item) for item in v] for k, v in raw_object.items()}


def get_minimum_and_maximum_height(boxes, iou):
    min_height = float('inf')
    max_height = float('-inf')

    for frame_boxes in boxes.values():
        for box in frame_boxes:
            if box.iou >= iou:
                min_height = min(min_height, box.height)
                max_height = max(max_height, box.height)
    return min_height, max_height


def create_historgram(boxes, iou_threshold):
    iou_values = []

    for frame_boxes in boxes.values():
        for box in frame_boxes:
            if box.iou >= iou_threshold:
                iou_values.append(box.iou)

    plt.hist(iou_values)
    plt.xlabel('IOU')
    plt.ylabel('Occurrences')
    plt.title('IOU Values Histogram')
    plt.show()


def calculate_average_iou(boxes: dict, iou):
    iou_sum = 0
    count = 0
    for frame in boxes.values():
        for box in frame:
            if box.iou >= iou:
                count += 1
                iou_sum += box.iou
    if count != 0:
        iou_avg = iou_sum / count
        return iou_avg
    return -1


def calculate_iou_for_2_boxes(box1, box2):
    # Calculate the coordinates of the intersection rectangle
    x1 = max(box1.x_center - box1.width / 2, box2.x_center - box2.width / 2)
    y1 = max(box1.y_center - box1.height / 2, box2.y_center - box2.height / 2)
    x2 = min(box1.x_center + box1.width / 2, box2.x_center + box2.width / 2)
    y2 = min(box1.y_center + box1.height / 2, box2.y_center + box2.height / 2)

    # Calculate the width and height of the intersection rectangle
    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    # Calculate the area of the intersection rectangle
    intersection_area = intersection_width * intersection_height

    area_of_overlap = (box1.width * box1.height + box2.width * box2.height) - intersection_area

    iou = intersection_area / area_of_overlap

    return iou


def make_data_dicts(gt_path: str, det_path: str) -> tuple[dict, dict]:
    """
    function receives paths of the GT retults tsv and the detection results tsv,
    and returns a dictionary for each one with the data loaded in to it from each file.
    dictionary format: {frame: [BBox, BBox, ...]}
    """
    # Load GT and detection results files
    gt_df = pd.read_csv(gt_path, delimiter="\t")
    detection_df = pd.read_csv(det_path, delimiter="\t")

    # Extract relevant information from the dataframes
    gt_boxes = {}
    detection_boxes = {}

    for _, row in gt_df.iterrows():
        frame_name = str(row['name'])
        if frame_name not in gt_boxes:
            gt_boxes[frame_name] = []
        gt_boxes[frame_name].append(BoundingBox(x_center=row['x_center'], y_center=row['y_center'], width=row['width'], height=row['height'], label=row['label']))

    for _, row in detection_df.iterrows():
        frame_name = str(row['name'])
        if frame_name not in detection_boxes:
            detection_boxes[frame_name] = []
        detection_boxes[frame_name].append(BoundingBox(x_center=row['x_center'], y_center=row['y_center'], width=row['width'], height=row['height']))
    
    return gt_boxes, detection_boxes


def set_ious(gt_boxes: dict, detection_boxes: dict, iou_threshold: float) -> None:
    """
    function receives GT data and detection data and sets the ious of each detection bbox.
    """
    for name, detection_bounding_box_list in detection_boxes.items():
        ground_truth_bounding_box_list = gt_boxes[name]
        for det_box in detection_bounding_box_list:
            for gt_bbox in ground_truth_bounding_box_list:
                iou = calculate_iou_for_2_boxes(det_box, gt_bbox)
                # saving the highest iou for a detection bounding box
                if iou > getattr(det_box, 'iou', 0):
                    det_box.iou = iou
                    det_box.label = gt_bbox.label
            if det_box.iou < iou_threshold:
                det_box.iou = 0
                det_box.label = 'None'


def find_unmatched_gts(gt_boxes: dict, detection_boxes: dict) -> dict:
    """
    function receives GT data and detection data and returns a dictionary with the GT boxes 
    in each frame that were unmatched to any detection boxes in the same frame.
    dictionary format: {frame: [BBox, BBox, ...]}
    """
    unmatched_gts = {}
    for frame, gt_boxes in gt_boxes.items():
        if frame in detection_boxes:
            det_boxes = detection_boxes[frame]
            for gt_box in gt_boxes:
                is_matched = False
                for det_box in det_boxes:
                    if det_box.iou > 0:
                        iou = calculate_iou_for_2_boxes(gt_box, det_box)
                        if iou == det_box.iou:
                            is_matched = True
                            break
                if not is_matched:
                    if frame not in unmatched_gts:
                        unmatched_gts[frame] = []
                    unmatched_gts[frame].append(gt_box)
    return unmatched_gts


def get_frames_with_labels(gt_boxes: dict, labels: list) -> list:
    """
    function receives GT data and a list of labels, and returns a list of frames
    that have at least one of the labels in them.
    """
    frames = []
    for frame, boxes in gt_boxes.items():
        for box in boxes:
            if box.label in labels:
                frames.append(frame)
                break
    return frames


def get_data_of_frames(boxes: dict, frames: list) -> dict:
    """
    function receieves boxes data and a list of frames and returns a
    dictionary with only data from the frames in the given list.
    dictionary format: {frame: [BBox, BBox, ...]}
    """
    frame_data = {}
    for frame in frames:
        if frame in gt_boxes:
            frame_data[frame] = boxes[frame]
    return frame_data


def find_unmatched_detection_boxes(detection_boxes: dict) -> list:
    """
    function receives detections data and returns a list of frames that have boxes
    that we not matched with any GT boxes.
    """
    frames_det_box_with_no_match = []
    for frame, boxes in detection_boxes.items():
        for box in boxes:
            if box.iou == 0:
                frames_det_box_with_no_match.append(frame)
                break
    return frames_det_box_with_no_match


def find_unmatched_gts_by_labels(gt_boxes: dict, detection_boxes: dict) -> dict:
    """
    function receives GT data and detections data, and returns a dictionary that has per label,
    all of the GT boxes that were not matched to any detection box.
    dictionary foramt: {label: [BBox, BBox, ...]}
    """
    unmatched_by_label = {}
    for frame, gt_boxes in gt_boxes.items():
        if frame in detection_boxes:
            det_boxes = detection_boxes[frame]
            for gt_box in gt_boxes:
                is_matched = False
                for det_box in det_boxes:
                    if det_box.iou > 0:
                        iou = calculate_iou_for_2_boxes(gt_box, det_box)
                        if iou == det_box.iou:
                            is_matched = True
                            break
                if not is_matched:
                    if gt_box.label not in unmatched_by_label:
                        unmatched_by_label[gt_box.label] = []
                    unmatched_by_label[gt_box.label].append(gt_box)
    return unmatched_by_label


def frame_iou_avg(detection_boxes: dict) -> None:
    """
    function receives detections data, and prints per frame the average IOU for that frame
    and how many boxes did not match with any GT box
    """
    for frame, boxes in detection_boxes.items():
        sum = 0
        count = 0
        zeros = 0
        for box in boxes:
            if box.iou == 0:
                zeros += 1
            else:
                sum += box.iou
                count += 1
        if count != 0:
            frame_avg = sum / count
            print(f"Frame: {frame}, AVG: {frame_avg}, No detections: {zeros}")
        else:
            print(f"Frame: {frame}, AVG: - , No detections: {zeros}")


def get_gt_label_counts(gt_boxes: dict) -> dict:
    """
    function receives GT data, and returns a dictionary of how many boxes there are with each label.
    dictionary format: {label: count}
    """
    gt_labels = {}
    for _, boxes in gt_boxes.items():
        for box in boxes:
            if box.label not in gt_labels:
                gt_labels[box.label] = 0
            gt_labels[box.label] += 1
    return gt_labels


def get_labels_per_frame(gt_boxes: dict) -> dict:
    """
    function receives GT data, and returns a dictionary of how many boxes there are
    with each label per frame.
    dictionary format: {frame: {label: count}}
    """
    labels_per_frame = {}
    for frame, boxes in gt_boxes.items():
        for box in boxes:
            if frame not in labels_per_frame:
                labels_per_frame[frame] = {}
            if box.label not in labels_per_frame[frame]:
                labels_per_frame[frame][box.label] = 0
            labels_per_frame[frame][box.label] += 1
    return labels_per_frame


def get_labels_per_frames(gt_boxes: dict, frames: list) -> dict:
    """
    function receives GT data and a list of frames, and returns a dictionary of how many
    boxes there are with each label per frame from the list of given frames.
    dictionary format: {frame: {label: count}}
    """
    frame_labels = {}
    for frame in frames:
        frame_boxes = gt_boxes[frame]
        for box in frame_boxes:
            if frame not in frame_labels:
                frame_labels[frame] = {}
            if box.label not in frame_labels[frame]:
                frame_labels[frame][box.label] = 0
            frame_labels[frame][box.label] += 1
    return frame_labels


def avg_iou_for_label(detection_boxes: dict, label: str) -> float:
    """
    function receives detections data and a label, and returns the average IOU of the boxes
    that have that label from all of the frames.
    """
    sum = 0
    count = 0
    for _, boxes in detection_boxes.items():
        for box in boxes:
            if box.label == label:
                sum += box.iou
                count += 1
    if count == 0:
        return -1
    else:
        return sum / count


def copy_images(source_folder, destination_folder, frames):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Iterate through files in the source folder
    for frame in frames:
        frame_name = f"AMBARELLA_3840_1920_{frame}.png"
        source_file_path = os.path.join(source_folder, frame_name)
        
        # Check if the file is a picture (you might need to adjust this condition)
        if os.path.isfile(source_file_path):
            destination_file_path = os.path.join(destination_folder, frame_name)
            # Copy the file to the destination folder
            shutil.copyfile(source_file_path, destination_file_path)


def plot_image_with_boxes(image_path, gt_bbox_list, det_bbox_list):
    # Open image
    img = Image.open(image_path)
    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(img)
    # Add bounding boxes
    for bbox in gt_bbox_list:
        # Calculate bounding box coordinates
        x_min = bbox.x_center - bbox.width / 2
        y_min = bbox.y_center - bbox.height / 2
        # Create a Rectangle patch
        rect = patches.Rectangle((x_min, y_min), bbox.width, bbox.height, linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
    # Add bounding boxes
    for bbox in det_bbox_list:
        # Calculate bounding box coordinates
        x_min = bbox.x_center - bbox.width / 2
        y_min = bbox.y_center - bbox.height / 2
        # Create a Rectangle patch
        rect = patches.Rectangle((x_min, y_min), bbox.width, bbox.height, linewidth=1, edgecolor='b', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
    # Show plot
    plt.show()


def plot_frame_with_unmatched_gt_det(frame: str, unmatched_gt_boxes: dict, detection_boxes: dict) -> None:
    unmatched_det_boxes = []
    for box in detection_boxes[frame]:
        if box.iou == 0:
            unmatched_det_boxes.append(box)
    plot_image_with_boxes(f"Q2_images/AMBARELLA_3840_1920_{frame}.png", unmatched_gt_boxes[frame], unmatched_det_boxes)


if __name__ == '__main__':

    gt_path = "highend_results.tsv"
    det_path = "detection_results.tsv"
    # Create GT and detections dictionaries of data from the tsv files
    gt_boxes, detection_boxes = make_data_dicts(gt_path, det_path)

    print("Total frames:", len(gt_boxes))

    # Define labels that are considered edge cases
    edge_case_labels = ['TRACTOR', 'KICK_SCOOTER', 'TRAILER', 'BIKE', 'CEMENT_MIXER_TRUCK', 'MOTOR', 'TRAIN', 'RIDER']

    # Get the frames that have an edge case label in them
    edge_frames = get_frames_with_labels(gt_boxes, edge_case_labels)
    print("Edge frames amount:", len(edge_frames))

    # Get the GT data and detections data of the frames that have edge case labels in them
    gt_boxes_edge_cases = get_data_of_frames(gt_boxes, edge_frames)
    detection_boxes_edge_cases = get_data_of_frames(detection_boxes, edge_frames)

    # Set the threshold of an IOU match between a GT box and a detection box
    iou_threshold = 0.6
    print(f"Chosen IOU threshold: {iou_threshold}")

    # Set the IOUs of the detection boxes according to the IOU threshold
    set_ious(gt_boxes_edge_cases, detection_boxes_edge_cases, iou_threshold)

    # Find frames that have GT boxes that werent matched to any detection boxes
    unmatched_gts = find_unmatched_gts(gt_boxes_edge_cases, detection_boxes_edge_cases)
    print("Unmatched GT pics count:", len(unmatched_gts))

    # Get the list of frames that have GT boxes that werent matched to any detection boxes
    frames_gt_box_with_no_match = list(unmatched_gts.keys())

    # Get the list of frames that have detection boxes that werent matched to any GT boxes
    frames_det_box_with_no_match = find_unmatched_detection_boxes(detection_boxes_edge_cases)
    print("Unmatched Detection pics amount:", len(frames_det_box_with_no_match))

    # Get a list of the frames that are the union of the frames that have GT boxes that werent matched
    # and the frames that have detection boxes that werent matched
    frames_to_use = list(set(frames_gt_box_with_no_match).union(frames_det_box_with_no_match))
    print("New total of union:", len(frames_to_use))

    # Get the detections data of the frames that were selected to be used
    data_to_use = get_data_of_frames(detection_boxes, frames_to_use)

    print("\n************* CHOSEN FRAMES DATA STATS *************")

    print(f"Amount of pics chosen: {len(data_to_use)}")

    min_height, max_height = get_minimum_and_maximum_height(data_to_use, iou_threshold)
    print(f"min height: {min_height}, max height: {max_height}")

    avg_iou = calculate_average_iou(data_to_use, iou_threshold)
    print(f"AVG IOU: {avg_iou}")

    # Copy chosen frames to a new folder
    # source_images = 'Q2_images'
    # chosen_frames = 'Q2_chosen_images'
    # copy_images(source_images, chosen_frames, frames_to_use)
    # print(f"Transfered chosen frames to {chosen_frames}")

    # Write the names of the chosen frames to a text file
    chosen_frames_txt = "Q2_chosen_frames.txt"
    with open(chosen_frames_txt, 'w') as file:
        for frame in frames_to_use:
            file.write(f"AMBARELLA_3840_1920_{frame}.png\n")

    print()
    create_historgram(data_to_use, iou_threshold)
