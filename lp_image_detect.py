import os
import cv2
import random
import colorsys
import numpy as np
import tensorflow as tf

from pydarknet import Detector
from pydarknet import Image as PDImage
from LPDetection.AI_utils.label import Label, dknet_label_conversion
from LPDetection.AI_utils.utils import crop_region, im2single, nms
from tensorflow.python.saved_model import tag_constants
from PIL import Image

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
input_size = 416

# config tensorflow in with GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

CONFIGURATION = {
    # vehicle detector module
    'VEHICLE_CFG': './Models/vehicle-detector/yolo-voc.cfg',
    'VEHICLE_WEIGHTS':'./Models/vehicle-detector/yolo-voc.weights',
    'VEHICLE_DATA': './Models/vehicle-detector/voc.data',
    # lp detector module
    'LP_MODEL': './Models/lp-detector/wpod-net_update1.h5',
    # ocr module
    'OCR_CFG': './Models/ocr/ocr-net.cfg',
    'OCR_WEIGHTS':'./Models/ocr/ocr-net.weights',
    'OCR_DATA': './Models/ocr/ocr-net.data',
    # Others configuration
    'IMG_WIDTH': 240,
    'IMG_HEIGHT': 120,
    'THRESHOLD_VEHICLE': 0.5,
    'THRESHOLD_LP': 0.5,
    'THRESHOLD_OCR': 0.3,
}
# load models
ocr_detector = Detector(bytes(CONFIGURATION['OCR_CFG'], encoding="utf-8"),
                             bytes(CONFIGURATION['OCR_WEIGHTS'], encoding="utf-8"), 0,
                             bytes(CONFIGURATION['OCR_DATA'],encoding="utf-8"))

# --- Detect vehicles and its lp ---
saved_model_loaded = tf.saved_model.load('./checkpoints/custom-416', tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']


def _recognize_lp(_lp_imgs):
    lp_values = []
    for _lp_img in _lp_imgs:
        _lp_I_img = PDImage(_lp_img)
        _lp_labels = ocr_detector.detect(_lp_I_img, thresh=CONFIGURATION['THRESHOLD_OCR'])

        if len(_lp_labels):
            _lp_labels = dknet_label_conversion(_lp_labels, CONFIGURATION['IMG_WIDTH'], CONFIGURATION['IMG_HEIGHT'])
            _lp_labels = nms(_lp_labels, .45)

            _y_value = [_label.tl()[1] for _label in _lp_labels]
            _y_mean = np.mean(_y_value)
            _y_var = np.var(_y_value)

            _lp_labels.sort(key=lambda x: x.tl()[0] + 1 if (x.tl()[1] > _y_mean) and (_y_var > 0.01) else x.tl()[0])
            if len(_lp_labels) > 4:
                _lp_str = ''.join([chr(l.cl()) for l in _lp_labels])
                lp_values.append(_lp_str)

    return lp_values

# helper function to convert bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
def format_boxes(bboxes, image_height, image_width):
    for box in bboxes:
        ymin = int(box[0] * image_height)
        xmin = int(box[1] * image_width)
        ymax = int(box[2] * image_height)
        xmax = int(box[3] * image_width)
        box[0], box[1], box[2], box[3] = xmin, ymin, xmax, ymax
    return bboxes

def draw_bbox(image, bboxes):
    bbox_color = (255, 0, 0)
    image_h, image_w, _ = image.shape

    out_boxes, out_scores, out_classes, num_boxes = bboxes
    for i in range(num_boxes):
        coor = out_boxes[i]

        height_ratio = int(image_h / 25)
        # separate coordinates from box
        xmin, ymin, xmax, ymax = coor
        # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
        pad = -8
        box = image[int(ymin) - pad:int(ymax) + pad, int(xmin) - pad:int(xmax) + pad]
        box = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
        box = cv2.cvtColor(box, cv2.COLOR_GRAY2BGR)
        box = cv2.resize(box, (240, 120))
        # read plate
        lp_values = _recognize_lp(_lp_imgs=[box])
        if lp_values:
            print('plate_number >>>>>', lp_values[0])
            cv2.putText(image, lp_values[0], (int(coor[0]), int(coor[1] - height_ratio)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)

        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
    return image

def recognize_lp_from_image(input_img):
    # change image color and size
    original_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.

    # add image into array
    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    batch_data = tf.constant(images_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    # run non max suppression on detections
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold=0.50
    )

    # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
    original_h, original_w, _ = original_image.shape
    bboxes = format_boxes(boxes.numpy()[0], original_h, original_w)

    # hold all detection data in one variable
    pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

    # read plate and draw lp box
    image = draw_bbox(original_image, pred_bbox)
    # save image result
    image = Image.fromarray(image.astype(np.uint8))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    return image
