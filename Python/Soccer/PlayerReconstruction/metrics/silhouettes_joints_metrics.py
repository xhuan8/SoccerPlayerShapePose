import torch
import os
import numpy as np

def compute_silh_error_metrics(pred_silhouettes, target_silhouettes, generate_vis=False):
    """
    Compute number of TP, FP, TN, FN between a batch of target silhouettes and
    predicted silhouettes.
    """
    pred_silhouettes = np.round(pred_silhouettes).astype(np.bool)
    target_silhouettes = np.round(target_silhouettes).astype(np.bool)

    true_positive = np.logical_and(pred_silhouettes, target_silhouettes)
    false_positive = np.logical_and(pred_silhouettes, np.logical_not(target_silhouettes))
    true_negative = np.logical_and(np.logical_not(pred_silhouettes),
                                   np.logical_not(target_silhouettes))
    false_negative = np.logical_and(np.logical_not(pred_silhouettes), target_silhouettes)

    num_tp = int(np.sum(true_positive))
    num_fp = int(np.sum(false_positive))
    num_tn = int(np.sum(true_negative))
    num_fn = int(np.sum(false_negative))

    global_acc = (num_tp + num_tn) / (num_tp + num_tn + num_fp + num_fn)
    iou = num_tp / (num_tp + num_fp + num_fn)
    precision = num_tp / (num_tp + num_fp + 1e-9)
    recall = num_tp / (num_tp + num_fn + 1e-9)
    f1 = (2 * precision * recall) / (precision + recall + 1e-9)

    result_vis = None
    if generate_vis:
        result_vis = np.zeros_like(pred_silhouettes).astype(np.uint8)
        result_vis += pred_silhouettes.astype(np.uint8) * 128
        result_vis += target_silhouettes.astype(np.uint8) * 64

    return {'global_acc': global_acc, 'iou': iou, 'f1': f1, 'precision': precision,
            'recall': recall}, result_vis


def compute_j2d_mean_l2_pixel_error(pred_joints2d, target_joints2d):
    l2_errors = np.linalg.norm(pred_joints2d - target_joints2d, axis=-1)
    mean_l2_pixel_error = np.mean(l2_errors)

    return mean_l2_pixel_error