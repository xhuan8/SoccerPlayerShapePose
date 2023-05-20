import cv2
import os
import shutil
import random
import time
import random
import sys
import timeit
import traceback
import uuid
import numpy as np
from sklearn.cluster import KMeans

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from global_var import *
from global_utils import *

class FieldDetector:
    def __init__(self):
        pass

    '''
    Detect the field of soccer game.
    Returns a mask with the same size as the input image, 
    pixel value 255 represents the field.
    '''
    def detect(self, image, filename, result_folder, is_save_log=False):
        starttime = timeit.default_timer()
        if (is_save_log):
            result_filename = os.path.join(result_folder, filename)
            #cv2.imwrite(result_filename.replace('.bmp', '_b.bmp'),
            #image[:,:,0])
            #cv2.imwrite(result_filename.replace('.bmp', '_g.bmp'),
            #image[:,:,1])
            #cv2.imwrite(result_filename.replace('.bmp', '_r.bmp'),
            #image[:,:,2])
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #if (is_save_log):
            #cv2.imwrite(result_filename.replace('.bmp', '_h.bmp'), hsv_image[:,:,0])
            #cv2.imwrite(result_filename.replace('.bmp', '_s.bmp'),
            #hsv_image[:,:,1])
            #cv2.imwrite(result_filename.replace('.bmp', '_v.bmp'),
            #hsv_image[:,:,2])
        
        hue = hsv_image[:,:,0]
        # kmeans takes too long time to process
        #data = hue.reshape((hue.shape[0] * hue.shape[1], 1))
        #kmeans = KMeans(n_clusters = 5)
        #kmeans.fit(data)
        #colors = kmeans.cluster_centers_
        #labels = kmeans.labels_
        #if (is_save_log):
            #print(colors)

        hue_hist = cv2.calcHist([hsv_image], [0], None, [180], (0, 181), accumulate=False)
        min_val, max_val, _, max_hue = cv2.minMaxLoc(hue_hist)
        #print('max_v: {}, max_pt: {}'.format(max_v, max_pt))
        hue_thresh = cv2.inRange(hue, max_hue[1] - 5, max_hue[1] + 15)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
        hue_open = cv2.morphologyEx(hue_thresh, cv2.MORPH_OPEN, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60))
        hue_dilate = cv2.morphologyEx(hue_open, cv2.MORPH_DILATE, kernel)
        largest = largest_connected_components(hue_dilate)
        hue_close = cv2.morphologyEx(largest, cv2.MORPH_CLOSE, kernel)

        contours, hierarchy = cv2.findContours(hue_close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #epsilon = 0.005*cv2.arcLength(contours[0],True)
        #approx = cv2.approxPolyDP(contours[0],epsilon,True)
        approx = cv2.convexHull(contours[0])
        result = np.zeros_like(hue_close)
        cv2.fillPoly(result, [approx], 255)
        if (is_save_log):
            #cv2.imwrite(result_filename.replace('.bmp', '_thresh.bmp'),
            #hue_thresh)
            #cv2.imwrite(result_filename.replace('.bmp', '_thresh1.bmp'),
            #hue_open)
            #cv2.imwrite(result_filename.replace('.bmp', '_thresh2.bmp'),
            #hue_dilate)
            #cv2.imwrite(result_filename.replace('.bmp', '_largest.bmp'),
            #largest)
            cv2.imwrite(result_filename.replace('.bmp', '_close.bmp'), hue_close)
            cv2.drawContours(image, [approx], 0, (0,0,255), 3)
            cv2.imwrite(result_filename.replace('.bmp', '_contours.bmp'), image)

        endtime = timeit.default_timer()
        print('time: {:.3f}'.format(endtime - starttime))
        return result


def test_single_image():
    filename_full = os.path.join(field_detection_data, 'Data/9c439d0c-7548-408a-8db4-009acd636fab.bmp')
    image = cv2.imread(filename_full)
    filename = os.path.basename(filename_full)

    result_folder = os.path.join(field_detection_data, 'Result')
    remake_dir(result_folder)

    orig_filename = os.path.join(result_folder, filename)
    #cv2.imwrite(orig_filename, image)

    detector = FieldDetector()
    result = detector.detect(image, filename, result_folder, True)
    if (result is not None):
        cv2.imwrite(orig_filename.replace('.bmp', '_result.bmp'), result)

#test_single_image()
def test_images():
    folder = os.path.join(field_detection_data, 'Data')
    files = os.listdir(folder)
    result_folder = os.path.join(field_detection_data, 'Result')
    remake_dir(result_folder)
    for filename in files:
        filename_full = os.path.join(folder, filename)
        image = cv2.imread(filename_full)

        orig_filename = os.path.join(result_folder, filename)
        #cv2.imwrite(orig_filename, image)

        detector = FieldDetector()
        result = detector.detect(image, filename, result_folder, True)
        if (result is not None):
            cv2.imwrite(orig_filename.replace('.bmp', '_result.bmp'), result)

test_images()
