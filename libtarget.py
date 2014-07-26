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

        # if len(hull) > 5:
        #     newcnt = []
        #     for d in defects:
        #         if d[0][3] < 500:
        #             start = d[0][0]
        #             end = d[0][1]
        #             for i in range(start, end):
        #                 newcnt.append(cnt[i])
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

        #
        # Make some plots
        #
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
