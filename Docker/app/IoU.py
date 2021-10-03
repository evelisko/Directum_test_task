import numpy as np
import tensorflow as tf


def loss(gt, pred):
    intersections = 0
    unions = 0
    diff_width = np.minimum(gt[:, 0] + gt[:, 2], pred[:, 0] + pred[:, 2]) - np.maximum(gt[:, 0], pred[:, 2])
    diff_height = np.minimum(gt[:, 1] + gt[:, 3], pred[:, 1] + pred[:, 3]) - np.maximum(gt[:, 1], pred[:, 3])
    intersection = diff_width * diff_height

    # Compute union
    area_gt = gt[:, 2] * gt[:, 3]
    area_pred = pred[:, 2] * pred[:, 3]
    union = area_gt + area_pred - intersection

    # Compute intersection and union over multiple boxes
    for j, _ in enumerate(union):
        if union[j] > 0 and intersection[j] > 0 and union[j] >= intersection[j]:
            intersections += intersection[j]
            unions += union[j]

    # Compute IOU. Use 1e-8 to prevent division by zero
    iou = np.round(intersections / (unions + 1e-8), 4)
    iou = iou.astype(np.float32)
    return iou


def IoU(y_true, y_pred):
    iou = tf.py_function(loss, [y_true, y_pred], tf.float32)
    return iou
