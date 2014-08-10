import copy

import scipy.spatial
import numpy as np
import cv2

from Target import Ellipse as _Ellipse
from Target import Square

class TargetDetectorBase(object):
    '''
    Generic Target Detector Class

    Inherit from this to write a new detector

    '''

    def __init__(self, image=None):
        if image:
            self.image = cv2.imread(image, 0)
        self.targets = None

    def find_targets(self, image):
        '''
        Routine that finds the targets
        '''
        if self.targets:
            return self.find_targets

    def get_threshold(self, d=5, sigmaColor=75, sigmaSpace=75, size=21, C=2):
        '''
        Returns the threshold of the image.
        Inputs:
        d, sigmaColor, sigmaSpace:
            See http://docs.opencv.org/modules/imgproc/doc/filtering.html?highlight=bilateralfilter#bilateralfilter
        size, C:
            see: http://docs.opencv.org/modules/imgproc/doc/miscellaneous_transformations.html?highlight=adaptivethreshold#cv2.adaptiveThreshold

        Returns threshold image with value 255 above threshold and 0 below
        '''
        blur = cv2.bilateralFilter(self.image, d, sigmaColor, sigmaSpace)
        thresh = cv2.adaptiveThreshold(blur, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, size, C)
        return thresh

    def get_contours(self):
        ''' Get contours in given image'''
        threshold = copy.deepcopy(self.threshold)
        contours, hierarchy = cv2.findContours(threshold,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        ch = zip(contours, hierarchy[0])
        # filter out short ones and keep inner ones only
        contours = [cnt for cnt in ch if len(cnt[0]) > 10]
        return contours
