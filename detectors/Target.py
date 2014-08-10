import cv2
import numpy as np


class Square(object):
    '''
        Object that represents a _projected_ square`
    '''

    def __init__(self, vertices):
        '''
            Initialise the object by giving it a list of vertices
            [[x1,y1],...[xn,yn]]
        '''

        self.vertices = vertices
        self._calculate_sides()
        self.children_ellipses = []

    def containsPoint(self, point):
        ''' Returns true if given point is inside this polygon

        '''
        return cv2.pointPolygonTest(self.vertices, point, False)

    def _calculate_sides(self):
        ''' Calculates the long side and short side length for the projected square

        '''
        v1, v2, v3, v4 = self.vertices

        # rough method but the poly will have 2 longer sides and two shorter
        # sides need to know more or less what they are

        side1 = (np.sum((v2-v1)**2))**0.5
        side2 = (np.sum((v3-v2)**2))**0.5
        side3 = (np.sum((v4-v3)**2))**0.5
        side4 = (np.sum((v1-v4)**2))**0.5

        temp1 = (side1 + side3)/2.0
        temp2 = (side2 + side4)/2.0

        self.longside = max(temp1, temp2)
        self.shortside = min(temp1, temp2)

    def contains(self, other):
        ''' Returns true if other's vertices fall inside self
        '''
        for ov in other.vertices:
            x, y = ov[0]
            if self.containsPoint((x, y)) > 0:
                return True
        return False

class Target(object):

    '''
    Represents a target.
    '''

    def __init__(self, x, y, type='default', encoding=None):
        self.x = x
        self.y = y
        self.type = type
        self.encoding = encoding


class Ellipse:
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
