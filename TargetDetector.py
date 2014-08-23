import copy

import scipy.spatial
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib

from Target import Ellipse as _Ellipse
from Target import Square


class TargetDetector(object):
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


class DefaultDetector(TargetDetector):

    '''
    Finds targets by calculating an adaptive thresholded image, on which it
    finds squares and ellipses from the contours on the thresholded image. From
    these lists it looks for targets that follow the Photomodeler encoding
    system and marks them as RAD targets and it looks for the smaller unencoded
    targets.

    returns nothing.

    self.radtargets contain the ellipses of the RAD targets
    self.smalltargets contains the ellipses of the unencoded targets

    # TODO fix this output to be consistent
    '''

    def __init__(self, image=None):
        super(DefaultDetector, self).__init__(image)
        self.detector_name = 'DefaultDetector'
        self.threshold = self.get_threshold()
        self.contours = self.get_contours()
        self.square_contours = self.find_square_contours()

    def _find_ellipses(self):
        ''' finds all the ellipses in the image
        '''
        ellipses = []
        hulls = []
        # for each contour, fit an ellipse
        for i, cnt in enumerate(self.contours):
            # get convex hull of contour
            hull = cv2.convexHull(cnt, returnPoints=True)
            # defects = cv2.convexityDefects(cnt, hull)

            if len(hull) > 5:
                # (x,y), (Ma, ma), angle = cv2.fitEllipse(hull)
                ellipse = cv2.fitEllipse(np.array(hull))
                ellipses.append(ellipse)
                hulls.append(hulls)

        return ellipses, hulls

    def find_targets(self):
        ''' Finds all the targets on itself
        '''
        # Find all the ellipses in the image. We are looking for the biggest
        # ellipse that fits inside on of the squares.

        ellipses, hulls = self._find_ellipses()
        self.ellipses = ellipses

        radtargets = []

        for ell in self.ellipses:
            (x, y), (Ma, ma), angle = ell
            Ma, ma = max(Ma, ma), min(Ma, ma)
            if Ma/ma > 3.5:
                continue
            if Ma < 15:
                continue
            outer_enc, inner_enc = self.find_rad_encoding(self.threshold, ell)
            if not (outer_enc == "011111111111"):
                continue
            if not inner_enc.startswith('1'):
                continue
            radtargets.append(((x, y), (ma, Ma), angle))

        self.radtargets = radtargets

        smalltargets = []
        for sq in self.square_contours:
            for ell in self.ellipses:
                (x, y), (Ma, ma), angle = ell
                Ma = max(Ma, ma)
                ma = min(Ma, ma)
                if sq.containsPoint((x, y)) > 0:
                    if sq.longside > Ma:
                        if sq.shortside > ma:
                            smalltargets.append(ell)

        small_target_kdtree = self._create_ellipse_kdtree(smalltargets)
        # now go through the rad targets and remove the smalltargets inside
        _to_remove = []
        for rad in self.radtargets:
            (x, y), (Ma, ma), angle = rad
            # find the small targets within one major axis from rad center
            nearest = small_target_kdtree.query_ball_point((x, y), Ma/2.)
            for n in nearest:
                _to_remove.append(smalltargets[n])

        for rem in _to_remove:
            try:
                smalltargets.remove(rem)
            except:
                pass

        self.smalltargets = smalltargets

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
        ''' Get contours in given image

        '''
        threshold = copy.deepcopy(self.threshold)
        contours, hierarchy = cv2.findContours(threshold,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        # filter out short ones
        contours = [cnt for cnt in contours if len(cnt) > 10]

        return contours

    def find_rad_encoding(self, img, radtarget, plot=False):
        ''' given an image and a rad target ellipse pair, find the encoding used

        return as a string containing 1 and 0 i.e. encoding='101010101010'

        '''
        (x, y), (Ma, ma), angle = radtarget
        outer = _Ellipse(x, y, 0.85*Ma, 0.85*ma, angle)
        inner = _Ellipse(x, y, 0.6*Ma, 0.6*ma, angle)

        pouter, theta_outer, imval_outer =\
            self.find_imval_at_ellipse_coordinates(img, outer, n=200)
        pinner, theta_inner, imval_inner =\
            self.find_imval_at_ellipse_coordinates(img, inner, n=200)

        try:
            # get the angles where the image value along the ellipse is zero
            theta_min = np.min(theta_outer[imval_outer == 0]*180/np.pi)

            # find the index of the smallest angle where this is true
            start = np.where(theta_outer*180/np.pi == theta_min)[0][0]

            # now roll the array so that it start at that index
            imval_outer = np.roll(imval_outer, -start)
            imval_outer_split = np.array_split(imval_outer, 12)
            imval_inner = np.roll(imval_inner, -start)
            imval_inner_split = np.array_split(imval_inner, 12)

            # now split that array into 12 nearly equally sized pieces
            # the median value should be either 255 or 0, calculate the
            # encoding
            for i, segment in enumerate(imval_outer_split):
                if np.median(segment) == 255:
                    imval_outer_split[i] = '1'
                else:
                    imval_outer_split[i] = '0'
            outer_enc = ''.join(imval_outer_split)
            # same for inner
            for i, segment in enumerate(imval_inner_split):
                if np.median(segment) == 255:
                    imval_inner_split[i] = '1'
                else:
                    imval_inner_split[i] = '0'
            inner_enc = ''.join(imval_inner_split)

        except ValueError:
            outer_enc, inner_enc = '999999999999', '999999999999'

        return outer_enc, inner_enc

    def _create_ellipse_kdtree(self, ellipses):
        ''' Given list of ellipses as return from _find_ellipses
            return the kd-tree made from their coordinates
        '''

        data = np.zeros((len(ellipses), 2), dtype='float')

        for i, ellipse in enumerate(ellipses):
            (x, y), (Ma, ma), angle = ellipse
            data[i, 0] = x
            data[i, 1] = y

        kdtree = scipy.spatial.KDTree(data)

        return kdtree

    def find_imval_at_ellipse_coordinates(self, img, ellipse, n=100):
        ''' Given rotated ellipse, return n coordinates along its perimeter '''
        # x=acos(theta) y=bsin(theta)
        theta = (ellipse.angle + 90)*np.pi/180.
        angles = np.linspace(0, 2*np.pi, n)
        # center of ellipse
        x0, y0 = ellipse.x, ellipse.y

        x = ellipse.Ma/2.0*np.cos(angles)
        y = ellipse.ma/2.0*np.sin(angles)

        xy = np.array([(x[i], y[i]) for i, xx in enumerate(x)]).T
        # rotation matrix
        rotMat = np.array([[np.sin(theta), -1.0*np.cos(theta)],
                           [np.cos(theta), np.sin(theta)]])

        rotatedXY = np.dot(rotMat, xy).T

        rotatedXY[:, 0] += y0
        rotatedXY[:, 1] += x0
        # round to ints
        rotatedXY = np.around(rotatedXY, 0)

        # find image values
        imval = []
        for row in rotatedXY:
            x, y = row[0], row[1]
            try:
                imval.append(img[x, y])
            except IndexError:
                imval.append(0)

        imval = np.array(imval)
        # regions are either high or low. make high regions = 255 and low
        # regions == 1.0
        imval_max = imval.max()

        imval[imval > 0.25*imval_max] = 255
        imval[imval <= 0.25*imval_max] = 0
        return rotatedXY, angles, np.array(imval)

    def find_square_contours(self, epsilon=0.1, min_area=200, max_area=4000):
        ''' Find the ones that is approximately square
        '''

        squares = []
        for cnt in self.contours:
            area = abs(cv2.contourArea(cnt))
            err = epsilon*cv2.arcLength(cnt, True)
            hull = cv2.convexHull(cnt)
            approx = cv2.approxPolyDP(hull, err, True)
            if len(approx) != 4:
                continue
            if area < min_area:
                continue
            if area > max_area:
                continue
            square = Square(approx)
            squares.append(square)

        return squares
