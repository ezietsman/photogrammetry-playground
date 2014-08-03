import copy

import cv2
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches

from Target import Square, ellipse


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

    def find_rad_encoding(self, img, radtarget, plot=False):
        ''' given an image and a rad target ellipse pair, find the encoding used

        return as a string containing 1 and 0 i.e. encoding='101010101010'

        '''
        (x, y), (Ma, ma), angle = radtarget
        ring = ellipse(x, y, Ma, ma, angle)

        pouter, theta_outer, imval_outer =\
            self.find_imval_at_ellipse_coordinates(img, ring, n=200)

        try:
            # get the angles where the image value along the ellipse is zero
            theta_min = np.min(theta_outer[imval_outer == 0]*180/np.pi)
            # find the index of the smallest angle where this is true
            start = np.where(theta_outer*180/np.pi == theta_min)[0][0]
            # now roll the array so that it start at that index
            imval_outer = np.roll(imval_outer, -start)
            imval_outer_split = np.array_split(imval_outer, 12)
            # now split that array into 12 nearly equally sized pieces
            # the median value should be either 255 or 0, calculate the encoding
            for i, segment in enumerate(imval_outer_split):
                if np.median(segment) == 255:
                    imval_outer_split[i] = '1'
                else:
                    imval_outer_split[i] = '0'
            encoding = ''.join(imval_outer_split)
        except ValueError as ve:
            #print ve
            encoding = '999999999999'

        # some bug fixing plots
        if plot:
            fig = plt.figure(figsize=(12, 12))
            ax1 = fig.add_subplot(111, aspect='equal')
            intMa = int(Ma)
            plt.imshow(img, cmap=matplotlib.cm.gray, interpolation='nearest')
            ell1 = radtarget
            e1 = matplotlib.patches.Ellipse((x, y), Ma, ma, angle,
                         facecolor='none', edgecolor='r')
#           # make ellipse for the inner encoding ring
#           e3 = Ellipse((ell2.x, ell2.y), ell2.Ma*(0.5), ell2.ma*(0.5),
#                        ell2.angle+90, facecolor='none', edgecolor='b')
#           # make ellipse for the outer encoding ring
#           e4 = Ellipse((ell2.x, ell2.y), ell2.Ma*(0.9), ell2.ma*(0.9),
#                        ell2.angle+90, facecolor='none', edgecolor='b')
            ax1.add_artist(e1)
#           ax1.add_artist(e2)
#           ax1.add_artist(e3)
#           ax1.add_artist(e4)
            plt.xlim(x-intMa, x+intMa)
            plt.ylim(y-intMa, y+intMa)
            plt.title(encoding)

            # TODO Roll the array so that the minimum of the outer ring is
            # at the start of the array, then calculate the values in the 12 bits

            plt.show()

        return encoding

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
            outer_ring = (x, y), (0.85*Ma, 0.85*ma), angle
            outer_enc = self.find_rad_encoding(self.threshold, outer_ring)
            if (outer_enc == "011111111111"):
                radtargets.append(ell)

        self.radtargets = radtargets

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

        MaMax = 100
        maMax = 100
        MaMin = 8
        maMin = 8
        Ma_ma = 5

        for i, ell in enumerate(ellipses):
            (x,y), (Ma, ma), angle = ell
            Ma = max(Ma, ma)/2.0
            ma = min(Ma, ma)/2.0

            if (Ma > MaMax) or (Ma < MaMin):
                del ellipses[i]
                del hulls[i]
                continue

            if (ma > maMax) or (ma < maMin):
                del ellipses[i]
                del hulls[i]
                continue

            if float(Ma) / float(ma) > Ma_ma:
                del ellipses[i]
                del hulls[i]
                continue

        return ellipses, hulls

    def get_threshold(self, d=8, sigmaColor=75, sigmaSpace=75, size=21, C=2):
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
