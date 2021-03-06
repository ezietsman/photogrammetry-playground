import itertools
import copy

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
import docopt

from libphotogrammetry.Image import Image



def get_contours(img):
    ''' Get contours in given image

    '''
    thresh = copy.deepcopy(img)
    contours, hierarchy = cv2.findContours(thresh,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    # filter out short ones
    contours = [cnt for cnt in contours if len(cnt) > 10]
    return contours


def get_threshold(img):
    blur = cv2.bilateralFilter(img, 5, 75, 75)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 21, 2)
    return thresh


def filter_ellipses(ellipses):
    ''' Filter out unlikely candidates from a list of ellipses

        Return the list of ellipses without the false postives
    '''

    # TODO Filter out
    #  weed out some spurious ellipses

    MaMax = 100
    maMax = 100
    MaMin = 8
    maMin = 8
    Ma_ma = 5

    for i, ell in enumerate(ellipses):
        Ma, ma = ell.Ma, ell.ma

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

    for ellipse1, ellipse2 in ellipse_pairs:
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




def find_rad_encoding(img, radtarget, plot=False):
    ''' given an image and a rad target ellipse pair, find the encoding used

    return as a string containing 1 and 0 i.e. encoding='101010101010'

    '''

    big = radtarget[1]

    outer = ellipse(big.x, big.y, big.Ma*0.85, big.ma*0.85, big.angle)
    inner = ellipse(big.x, big.y, big.Ma*0.6, big.ma*0.6, big.angle)

    pouter, theta_outer, imval_outer =\
        find_imval_at_ellipse_coordinates(img, outer, n=200)
    pinner, theta_inner, imval_inner =\
        find_imval_at_ellipse_coordinates(img, inner, n=200)

    try:
        # get the angles where the image value along the ellispe is zero
        theta_min = np.min(theta_outer[imval_outer == 0]*180/np.pi)
        # find the index of the smallest angle where this is true
        start = np.where(theta_outer*180/np.pi == theta_min)[0][0]
        # now roll the array so that it start at that index
        imval_inner = np.roll(imval_inner, -start)
        imval_inner_split = np.array_split(imval_inner, 12)
        # now split that array into 12 nearly equally sized pieces
        # the median value should be either 255 or 0, calculate the encoding
        for i, segment in enumerate(imval_inner_split):
            if np.median(segment) == 255:
                imval_inner_split[i] = '1'
            else:
                imval_inner_split[i] = '0'
        encoding = ''.join(imval_inner_split)
    except ValueError as ve:
        print ve
        encoding = '------------'
    print encoding

    # some bug fixing plots
    if plot:
        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(111, aspect='equal')
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
        plt.title(encoding)

        # TODO Roll the array so that the minimum of the outer ring is
        # at the start of the array, then calculate the values in the 12 bits

        plt.show()

    if encoding.startswith('0'):
        return False

    return encoding


def find_rad_targets2(img, ellipses):
    ''' Find rad targets using a different algorithm
    For every ellipse find the outer ring and see if it has one and only
    one gap in a ring of approx 85% the radius of the big ellipse. If so it
    is a radtarget.

    Doesn't seem to work that well... :(

    Return coordinates and encoding for each found rad target
    '''
    radtargets = []

    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(121, aspect='equal')
    plt.imshow(img, interpolation='nearest', cmap=cm.gray)

    for (x, y), (Ma, ma), angle in ellipses:
        outer = ellipse(x, y, Ma*0.85, ma*0.85, angle)
        pouter, theta_outer, imval_outer =\
            find_imval_at_ellipse_coordinates(img, outer, n=1000)

        total = imval_outer.max()*imval_outer.size
        if imval_outer.sum() < (total - total/12.)*1.05:
            if imval_outer.sum() > (total - total/12.)*0.95:
                e1 = Ellipse((outer.x, outer.y), outer.Ma, outer.ma,
                             outer.angle+90,
                             facecolor='none', edgecolor='r')
                ax1.add_artist(e1)

                inner = ellipse(outer.x, outer.y, outer.Ma*0.25,
                                outer.ma*0.25, outer.angle)
                radtargets.append((inner, outer))
    plt.show()
    return radtargets


__doc__ = '''\
Usage: findtargets.py JPG... [--plot] [--save]
'''

if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    for jpg in arguments['JPG']:
        print("Finding RAD targets in {f}".format(f=jpg))
        image = Image(jpg)
        image.find_targets()

        print("    found {n} RAD targets!".format(n=len(image.radtargets)))
        print("    found {n} Small targets!".format(n=len(image.smalltargets)))

        if arguments['--plot'] or arguments['--save']:
            fig = plt.figure(figsize=(12, 12))
            ax1 = fig.add_subplot(111, aspect='equal')
            plt.imshow(image.image, cmap=cm.gray, interpolation='nearest')

#           for sq in image.square_contours:
#               vertices = sq.vertices
#               vertices = np.append(vertices, vertices[0]).reshape((5,2))
#               plt.plot(vertices[:,0], vertices[:,1], 'b-', linewidth=3)

#           for ell in image.ellipses:
#               (x, y), (Ma, ma), angle = ell
#               ell = Ellipse([x, y], Ma, ma, angle,
#                             facecolor='none',
#                             edgecolor='y')
#               ax1.add_artist(ell)

            for ell in image.smalltargets:
                (x, y), (Ma, ma), angle = ell
                ell = Ellipse([x, y], Ma, ma, angle,
                              facecolor='none',
                              edgecolor='b',
                              linewidth=2)
                ax1.add_artist(ell)


            for ell in image.radtargets:
                (x, y), (Ma, ma), angle = ell
                ell = Ellipse([x, y], Ma, ma, angle,
                              facecolor='none',
                              edgecolor='r')
                ax1.add_artist(ell)

        if arguments['--save']:
            plt.savefig('test.png'.format(
                f=jpg.replace('.jpg', '')), bbox_inches='tight')

        if arguments['--plot']:
            plt.show()
