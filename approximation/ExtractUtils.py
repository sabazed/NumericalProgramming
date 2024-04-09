import cv2
import numpy as np
import urllib.request

class ExtractUtils:

    def __init__(self, radius, isLocal):
        self._radius = radius   # Radius for filtering points
        self._isLocal = isLocal # Determine where to take image from

    
    def _filter_points(self, points):
        res = list()
        cur = None
        while len(points) > 0:
            # Store next point
            cur = points[0]
            res.append(cur)
            # Filter out all other points which are closer than a predefined radius limit to the recently added point
            points = [p for p in points if np.linalg.norm(p - cur) > self._radius] 
        return res


    def get_curve_points(self, url):
        # Read the image
        img = None
        if (self._isLocal):
            img = cv2.imread(url)
        else:
            url_response = urllib.request.urlopen(url)
            img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
            img = cv2.imdecode(img_array, -1)
        
        max_val = 255
        tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tmp = cv2.GaussianBlur(tmp, (3, 3), cv2.BORDER_DEFAULT)
        tmp = cv2.Canny(tmp, 0, max_val)
        tmp = cv2.adaptiveThreshold(tmp, max_val, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)
        contours, h = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get contours' coords into a list
        points = [np.array([p.squeeze() for p in c]) for c in contours]       
        # Filter out excess points
        points = np.array(self._filter_points([p for parr in points for p in parr]))
        x, y = [], []
        for point in points:
          x.append(point[0])
          y.append(-point[1])
        # Reappend starting points to form a closed curve
        x.append(x[0])
        y.append(y[0])
        
        return x, y