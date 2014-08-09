from TargetDetectorBase import TargetDetectorBase

class PolyValueDetector(TargetDetectorBase):

    '''
    Finds targets by assessing the median threshold value close to ellipses

    '''

    def __init__(self, image=None):
        super(PolyValueDetector, self).__init__(image)
        self.detector_name = 'PolyValueDetector'
        self.threshold = self.get_threshold()
        self.contours = self.get_contours()
        self.ellipses, self.hulls = self._find_ellipses()
        self.smallpolys = self.find_square_contours()


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

    def _find_ellipses(self):
        ''' finds all the ellipses in the image
        '''
        ellipses = []
        hulls = []
        # for each contour, fit an ellipse
        for i, ch in enumerate(self.contours):
            cnt = ch[0]
            # get convex hull of contour
            hull = cv2.convexHull(cnt, returnPoints=True)
            # defects = cv2.convexityDefects(cnt, hull)

            if len(hull) > 5:
                # (x,y), (Ma, ma), angle = cv2.fitEllipse(hull)
                ellipse = cv2.fitEllipse(np.array(hull))
                (x, y), (Ma, ma), angle = ellipse
                # get rid of tiny ones
                if (max(Ma,ma) < 8) or (min(Ma,ma) < 8):
                    continue
                # get rid of large ones
                if (max(Ma,ma) > 80) or (min(Ma,ma) > 80):
                    continue
                ellipses.append(ellipse)
                hulls.append(hulls)

        return ellipses, hulls

    def find_square_contours(self, epsilon=0.1, min_area=200, max_area=4000):
        ''' Find the ones that is approximately square
        '''

        squares = []
        for cnt in self.contours:
            area = abs(cv2.contourArea(cnt))
            err = epsilon*cv2.arcLength(cnt, True)
            hull = cv2.convexHull(cnt)
            approx = cv2.approxPolyDP(hull, err, True)
            if len(approx) < 4:
                continue
            if len(approx) > 7:
                continue
            if area < min_area:
                continue
            if area > max_area:
                continue
            squares.append(approx)

        return squares

    def find_targets(self):
        '''

        '''
