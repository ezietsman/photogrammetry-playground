import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse

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

    ret,thresh = cv2.threshold(img, 127, 255, 0)

    return thresh


def filter_ellipses(ellipses):
    ''' Filter out unlikely candidates from a list of ellipses

        Return the list of ellipses without the false postives
    '''


    # TODO Filter out
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

    return ellipses

class ellipse:
    def __init__(self, x, y, Ma, ma, angle):
        self.x = x
        self.y = y
        self.Ma = Ma
        self.ma = ma
        self.angle = angle

    def isCloseTo(self, other, err=0.5):
        ''' Returns True if ellipse is within 0.5 pixels of given ellipse
        '''
        if abs(self.x - other.x) < err:
            if abs(self.y - other.y) < err:
                return True
        return False

    def hasSameRotation(self, other, err=0.0001):
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

    for i, ell1 in enumerate(ellipses):
        (x, y), (Ma, ma), angle = ell1
        ellipse1 = ellipse(x, y, Ma, ma, angle)
        for j, ell2 in enumerate(ellipses[i+1:]):
            (x, y), (Ma, ma), angle = ell1
            ellipse2 = ellipse(x, y, Ma, ma, angle)
            # now check if they are in fact 'concentric'
            if ellipse1.isConcentricTo(ellipse2):
                if ellipse1.isSmallerThan(ellipse2):
                    rad_targets.append([ellipse1, ellipse2])
                else:
                    rad_targets.append([ellipse2, ellipse1])
    return rad_targets

def find_ellipses(contours):

    ellipses = []
    hulls = []
    # for each contour, fit an ellipse
    for i, cnt in enumerate(contours):
        # get convex hull of contour
        hull = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull)

        if len(hull) > 5:
            newcnt = []
            for d in defects:
                if d[0][3] < 500:
                    start = d[0][0]
                    end = d[0][1]
                    for i in range(start, end):
                        newcnt.append(cnt[i])
            if len(newcnt) > 5: 
                #(x,y), (Ma, ma), angle = cv2.fitEllipse(hull)
                ellipse = cv2.fitEllipse(np.array(newcnt))
                (x,y), (Ma, ma), angle = ellipse
                
                ellipses.append(ellipse)
                hulls.append(hulls)
        
    return ellipses, hulls



def find_edges(img):
    ''' Find edges in the given image
    '''

    edges = cv2.Canny(img, 100, 200)

    return edges



if __name__ == "__main__":

    img = cv2.imread('radtargets/rad250.jpg', 0)
    thresh = get_threshold(img)
    edges = find_edges(thresh)
    contours = get_contours(edges)
    ellipses, hulls = find_ellipses(contours)
    radtargets = find_rad_targets(ellipses)
    print len(ellipses)
    print len(radtargets)

    fig = plt.figure(figsize=(12,12))
    fig.add_subplot(121, aspect='equal')
    plt.imshow(img, cmap=cm.gray, interpolation='nearest')

    for cnt in contours:
        cnt = np.array(cnt)
        x = cnt[:,0][:,0]
        y = cnt[:,0][:,1]

        np.append(x, x[0])
        np.append(y, y[0])

        plt.plot(x, y)

    ax = fig.add_subplot(122, aspect='equal')
    plt.imshow(edges, cmap=cm.gray, interpolation='nearest')
    
    for ellipse in ellipses:
        (x,y), (Ma, ma), angle = ellipse
        ell = Ellipse([x,y], Ma, ma, angle, facecolor='none', edgecolor='r')
        ax.add_artist(ell)

    plt.show()

