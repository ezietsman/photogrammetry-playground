import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse



def apply_threshold(img):

    smoothed = cv2.medianBlur(img, 3)
    threshold = cv2.adaptiveThreshold(smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,31,10)

    return threshold


def find_ellipses(img):

    ellipses = []
    hulls = []
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # for each contour, fit an ellipse
    for i, cnt in enumerate(contours):
        # get convex hull of contour
        hull = cv2.convexHull(cnt)

        if len(hull) > 5:
            #(x,y), (Ma, ma), angle = cv2.fitEllipse(hull)
            ellipse = cv2.fitEllipse(hull)
            (x,y), (Ma, ma), angle = ellipse
            
            ellipses.append(ellipse)
            hulls.append(hulls)
        

    #  weed out some spurious ellipses
    
    MaMax = 100
    maMax = 100
    MaMin = 4
    maMin = 4
    Ma_ma = 4

    for i, ell in enumerate(ellipses):
        Ma,ma = ell[1]

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

def prepare_image(img):
    
    prepared_image = img

    return 



def find_rad_targets(img):
    ''' Find all rad encoded targets in given img object

    return coordinates of center and number of encoding
    '''

    return 






if __name__ == "__main__":

#   img = cv2.imread('img_5661.jpg',0)
    img = cv2.imread('radcoded-targets/rad128.jpg', 0)
#   edges = cv2.Canny(img, 100, 500)

    fig = plt.figure()

    fig.add_subplot(121, aspect='equal')
    thresh = apply_threshold(img)

    ellipses, hulls = find_ellipses(thresh)

    plt.imshow(thresh, cmap=cm.gray, interpolation='nearest')

    ax = fig.add_subplot(122, aspect='equal')
    plt.imshow(img, interpolation='nearest', cmap=cm.gray)
    
    for ellipse in ellipses:
        (x,y), (Ma, ma), angle = ellipse
        ell = Ellipse([x,y], Ma, ma, angle, facecolor='none', edgecolor='r')
        ax.add_artist(ell)
    
    print len(hulls)
    plt.show()

