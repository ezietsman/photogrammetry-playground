import copy

import cv2
import numpy as np
import scipy.spatial

from Target import Square


class Image(object):

    '''
        This class represents one image that will be used in a photogrammetric
        calculation
    '''

    def __init__(self, filename):
        ''' Initialise the Image by giving it a filename '''
        self.filename = filename

        self.threshold = None
        self.contours = None
        self.square_contours = None
        self.targets = []

        self.image = cv2.imread(filename, 0)
        self.threshold = self.get_threshold()
        self.contours = self.get_contours()
        self.square_contours = self.find_square_contours()

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

    def find_RAD_targets(self):
        ''' Finds all the targets on itself
        '''

        # Find all the ellipses in the image. We are looking for the biggest
        # ellipse that fits inside on of the squares.

        ellipses, hulls = self._find_ellipses()
        self.ellipses = ellipses

        # going to use the kdtree to find the ellipses nearest to the squares
        kdtree = self._create_ellipse_kdtree(ellipses)
        for square in self.square_contours:
            # calculate the centroid
            M = cv2.moments(square.vertices)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            square_area = M['m00']
            nearest = kdtree.query_ball_point(np.array([cx, cy]), 20)
            # we want the biggest ellipse that is close AND smaller than square
            biggest_neighbour_size = 0.0
            biggest_neighbour = None
            for nth in nearest:
                (x, y), (Ma, ma), angle = ellipses[nth]
                size = np.pi*Ma*ma/4.0
                if (size > biggest_neighbour_size) & (size < square_area):
                    biggest_neighbour = nth
                    biggest_neighbour_size = size

            if not biggest_neighbour:
                continue



        print "Ellipses", len(ellipses)
        print "Squares", len(self.square_contours)
        print "Targets", len(self.targets)

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
        if self.threshold is not None:
            return self.threshold

        blur = cv2.bilateralFilter(self.image, d, sigmaColor, sigmaSpace)
        thresh = cv2.adaptiveThreshold(blur, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, size, C)
        return thresh

    def get_contours(self):
        ''' Get contours in given image

        '''
        if self.contours is not None:
            return self.contours

        threshold = copy.deepcopy(self.threshold)
        contours, hierarchy = cv2.findContours(threshold,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        # filter out short ones
        contours = [cnt for cnt in contours if len(cnt) > 10]

        return contours

    def find_square_contours(self, epsilon=0.05, min_area=200, max_area=4000):
        ''' Find the ones that is approximately square
        '''

        if self.square_contours is not None:
            return self.square_contours

        if self.contours is None:
            self.contours = self.get_contours()

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
