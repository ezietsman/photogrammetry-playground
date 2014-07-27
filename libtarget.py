import itertools

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
import docopt


class ellipse:
    def __init__(self, x, y, Ma, ma, angle):
        self.x = x
        self.y = y
        self.Ma = max(Ma, ma)
        self.ma = min(Ma, ma)
        self.angle = angle

    def isCloseTo(self, other, err=1):
        ''' Returns True if ellipse is within 0.5 pixels of given ellipse
        '''
        if abs(self.x - other.x) < err:
            if abs(self.y - other.y) < err:
                return True
        return False

    def hasSameRotation(self, other, err=10):
        ''' Returns True if ellipse is has the same rotation as other
        '''

        if abs(self.angle - other.angle) < err:
            return True

        return False

    def isSmallerThan(self, other):
        ''' Return True if both axes are smaller than other ellipse's '''
        if self.Ma < other.Ma:
            if self.ma < other.ma:
                return True

        return False

    def isConcentricTo(self, other):
        ''' similar position, and rotation
        '''
        if self.isCloseTo(other):
            if self.hasSameRotation(other):
                return True

        return False

    def __str__(self):
        _str = "x: {x}\ny: {y}\nMa: {Ma}\nma: {ma}\nangle: {ang}".format(
            x=self.x, y=self.y, Ma=self.Ma, ma=self.ma, ang=self.angle)
        return _str


def get_contours(img):
    ''' Get contours in given image

    '''
    contours, hierarchy = cv2.findContours(thresh,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    # filter out short ones
    contours = [cnt for cnt in contours if len(cnt) > 10]

    return contours


def get_threshold(img):
    blur = cv2.bilateralFilter(img, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 2)
    return thresh


def filter_ellipses(ellipses):
    ''' Filter out unlikely candidates from a list of ellipses

        Return the list of ellipses without the false postives
    '''

    # TODO Filter out
    #  weed out some spurious ellipses

    MaMax = 50
    maMax = 50
    MaMin = 4
    maMin = 3
    Ma_ma = 5

    for i, ell in enumerate(ellipses):
        Ma, ma = ell[1]

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

    return ellipses


def find_rad_targets(ellipses):
    '''
    Given a list of ellipses, return a list of rad-targets, specified by their
    inner ellipse and outer ellipse i.e.

    [
        [small_ellipse, big_ellipse]
        .
        .
        .
    ]
    '''
    rad_targets = []

    # photomodeler rad targets have as specific small/large circle ratio
    # for each ellipse, find the closest in position which is smaller in both
    # axes and has a similar angle

    ellipse_pairs = itertools.combinations(ellipses, 2)

    for ell1, ell2 in ellipse_pairs:
        (x1, y1), (Ma1, ma1), ang1 = ell1
        (x2, y2), (Ma2, ma2), ang2 = ell2
        ellipse1 = ellipse(x1, y1, Ma1, ma1, ang1)
        ellipse2 = ellipse(x2, y2, Ma2, ma2, ang2)
        # now check if they are in fact 'concentric'
        if ellipse1.isCloseTo(ellipse2):
            if ellipse1.isSmallerThan(ellipse2):
                if 3.5 < (ellipse2.Ma / ellipse1.Ma) < 5:
                    rad_targets.append((ellipse1, ellipse2))
            else:
                if 3.5 < (ellipse1.Ma / ellipse2.Ma) < 5:
                    rad_targets.append((ellipse2, ellipse1))

    return rad_targets


def find_ellipses(contours):

    ellipses = []
    hulls = []
    # for each contour, fit an ellipse
    for i, cnt in enumerate(contours):
        # get convex hull of contour
        hull = cv2.convexHull(cnt, returnPoints=True)
        # defects = cv2.convexityDefects(cnt, hull)

        if len(hull) > 5:
            # (x,y), (Ma, ma), angle = cv2.fitEllipse(hull)
            ellipse = cv2.fitEllipse(np.array(hull))
            ellipses.append(ellipse)
            hulls.append(hulls)

    return ellipses, hulls


def find_edges(img):
    ''' Find edges in the given image
    '''

    edges = cv2.Canny(img, 100, 255, 31)

    return edges


def find_imval_at_ellipse_coordinates(img, ellipse, n=100):
    ''' Given rotated ellipse, return n coordinates along its perimeter '''
    # x=acos(theta) y=bsin(theta)
    theta = (ellipse.angle + 90)*np.pi/180.
    angles = np.linspace(0, 2*np.pi, n)    
    # center of ellipse
    x0, y0 = ellipse.x, ellipse.y

    x = ellipse.Ma/2.0*np.cos(angles) 
    y = ellipse.ma/2.0*np.sin(angles) 
    
    xy = np.array([(x[i],y[i]) for i, xx in enumerate(x)]).T
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
        imval.append(img[x, y])
    imval = np.array(imval)
    # regions are either high or low. make high regions = 255 and low
    # regions == 1.0
    imval_max = imval.max()
    
    imval[imval > 0.25*imval_max] = 255
    imval[imval <= 0.25*imval_max] = 0
    print imval
    return rotatedXY, angles, np.array(imval)



def find_rad_encoding(img, radtarget):
    ''' given an image and a rad target ellipse pair, find the encoding used

    return as a string containing 1 and 0 i.e. encoding='101010101010'

    '''

    plot = True

    small = radtarget[0]
    big = radtarget[1]

    outer = ellipse(big.x, big.y, big.Ma*0.85, big.ma*0.85, big.angle)
    inner = ellipse(big.x, big.y, big.Ma*0.55, big.ma*0.55, big.angle)
    
    pouter, theta_outer, imval_outer = find_imval_at_ellipse_coordinates(img, outer, n=200)
    pinner, theta_inner, imval_inner = find_imval_at_ellipse_coordinates(img, inner, n=200)

    # some bug fixing plots
    if plot:
        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(211, aspect='equal')
        x, y = int(big.x), int(big.y)
        intMa = int(big.Ma)
        plt.imshow(img, cmap=cm.gray, interpolation='nearest')
        ell1, ell2 = radtarget
        e1 = Ellipse((ell1.x, ell1.y), ell1.Ma, ell1.ma, ell1.angle+90,
                     facecolor='none', edgecolor='r')
        e2 = Ellipse((ell2.x, ell2.y), ell2.Ma, ell2.ma, ell2.angle+90,
                     facecolor='none', edgecolor='r')
        # make ellipse for the inner encoding ring
        e3 = Ellipse((ell2.x, ell2.y), ell2.Ma*(0.5), ell2.ma*(0.5),
                     ell2.angle+90, facecolor='none', edgecolor='b')
        # make ellipse for the outer encoding ring
        e4 = Ellipse((ell2.x, ell2.y), ell2.Ma*(0.9), ell2.ma*(0.9),
                     ell2.angle+90, facecolor='none', edgecolor='b')
        ax1.add_artist(e1)
        ax1.add_artist(e2)
        ax1.add_artist(e3)
        ax1.add_artist(e4)
        plt.xlim(x-intMa, x+intMa)
        plt.ylim(y-intMa, y+intMa)

        plt.subplot(212)
        plt.plot(theta_inner*180/np.pi, imval_inner)
        plt.plot(theta_outer*180/np.pi, imval_outer)
        
        # middle of zero
        theta_mid = np.average(theta_outer[imval_outer == 0]*180/np.pi)
        plt.vlines(theta_mid, 0, 255, colors='r')

        # TODO Roll the array so that the minimum of the outer ring is 
        # at the start of the array, then calculate the values in the 12 bits


        plt.show()


__doc__ = '''\
Usage: libtargets.py JPG... [--plot] [--save]
'''

if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    for jpg in arguments['JPG']:
        print("Finding RAD targets in {f}".format(f=jpg))
        img = cv2.imread(jpg, 0)
        thresh = get_threshold(img)
        edges = find_edges(thresh)
        contours = get_contours(thresh)
        ellipses, hulls = find_ellipses(contours)
        radtargets = find_rad_targets(ellipses)
        print("    Found {N} rad-target candidates!".format(N=len(radtargets)))
        for rt in radtargets:
            find_rad_encoding(img, rt)
        #
        # Make some plots
        #
        if arguments['--plot'] or arguments['--save']:
            fig = plt.figure(figsize=(12, 12))
            ax1 = fig.add_subplot(121, aspect='equal')
            plt.imshow(img, cmap=cm.gray, interpolation='nearest')

            for rt in radtargets:
                ell1, ell2 = rt
                e1 = Ellipse((ell1.x, ell1.y), ell1.Ma, ell1.ma, ell1.angle+90,
                             facecolor='none', edgecolor='r')
                e2 = Ellipse((ell2.x, ell2.y), ell2.Ma, ell2.ma, ell2.angle+90,
                             facecolor='none', edgecolor='r')
                ax1.add_artist(e1)
                ax1.add_artist(e2)

            ax = fig.add_subplot(122, aspect='equal')
            plt.imshow(edges, cmap=cm.gray, interpolation='nearest',
                       vmin=0, vmax=55)

            for ell in ellipses:
                (x, y), (Ma, ma), angle = ell
                ell = Ellipse([x, y], Ma, ma, angle, facecolor='none',
                              edgecolor='r')
                ax.add_artist(ell)

        if arguments['--save']:
            plt.savefig('test.png'.format(
                f=jpg.replace('.jpg', '')), bbox_inches='tight')

        if arguments['--plot']:
            plt.show()
