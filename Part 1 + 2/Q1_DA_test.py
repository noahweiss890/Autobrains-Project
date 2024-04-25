import json
import matplotlib.pyplot as plt

class BoundingBox:
    def __init__(self, x_center=0, y_center=0, width=0, height=0, iou=0):
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height
        self.iou = iou


def read_lean_map_of_bboxes(input_json_file_path: str):
    with open(input_json_file_path, 'r') as json_stream:
        raw_object = json.load(json_stream)
    return {k: [BoundingBox(**item) for item in v] for k, v in raw_object.items()}


def get_minimum_and_maximum_height(boxes, iou):
    # Initialize variables to store minimum and maximum heights
    min_height = float('inf')
    max_height = float('-inf')

    # Iterate over frames and their boxes
    for frame_boxes in boxes.values():
        for box in frame_boxes:
            # Check if the box has an IOU value greater than or equal to the threshold
            if box.iou >= iou:
                # Update minimum and maximum heights if needed
                min_height = min(min_height, box.height)
                max_height = max(max_height, box.height)
    return min_height, max_height


def create_historgram(boxes, iou_threshold):
    # Initialize an empty list to store IOU values
    iou_values = []

    # Iterate over frames and their boxes
    for frame_boxes in boxes.values():
        for box in frame_boxes:
            # Check if the box has an IOU value greater than the threshold
            if box.iou >= iou_threshold:
                iou_values.append(box.iou)

    # Plot the histogram
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
    
    iou_avg = iou_sum / count

    return iou_avg


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


if __name__ == '__main__':
    path_detection_boxes_json = "Q1_system_output.json"
    path_groundtruth_boxes_json = "Q1_gt.json"
    iou_threshold = 0.5

    detection_boxes = read_lean_map_of_bboxes(path_detection_boxes_json)
    ground_truth_boxes = read_lean_map_of_bboxes(path_groundtruth_boxes_json)

    for name, detection_bounding_box_list in detection_boxes.items():
        ground_truth_bounding_box_list = ground_truth_boxes[name]
        for det_box in detection_bounding_box_list:
            for gt_bbox in ground_truth_bounding_box_list:
                iou = calculate_iou_for_2_boxes(det_box, gt_bbox)
                # saving the highest iou for a detection bounding box
                if iou > getattr(det_box, 'iou', 0):
                    det_box.iou = iou

    # calculate average iou for the boxes that pass > iou_threshold
    average_iou = calculate_average_iou(detection_boxes, iou_threshold)
    print(f"The average_iou is: {average_iou}")

    # create histogram for the boxes that pass iou_threshold.
    # x_axis: iou, y_axis: occurrences
    create_historgram(detection_boxes, iou_threshold)

    # find the minimum and the maximum height for the boxes that pass > iou_threshold
    min_height, max_height = get_minimum_and_maximum_height(detection_boxes, iou_threshold)
    print(f"The min_height is: {min_height} and the max_height is: {max_height}")
