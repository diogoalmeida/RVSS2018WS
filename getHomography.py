import cv2
import numpy as np
import urllib

IMG_DIM = (320, 240)
NUM_SAMPLES = 4
class getHomography:

    def __init__(self, image):
        self._img = image
        self._right_clicks = []
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('image', self.mouse_callback)
        cv2.imshow('image', self._img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def mouse_callback(self, event, x, y, flags, params):
        """Get the  image coordinates of a right mouse click."""
        if event == 1:
            #store the coordinates of the right-click event
            self._right_clicks.append([x, y])

            #this just verifies that the mouse data is being collected
            #you probably want to remove this later
            print self._right_clicks

            if len(self._right_clicks) >= NUM_SAMPLES:
                self.compute_homography()

    def compute_homography(self):
        """Get points values and compute homography."""
        l = []
        for i in range(1,NUM_SAMPLES + 1):
            s = raw_input('insert p' + str(i) + ': ')
            p = list(map(float, s.split()))
            l = l + [p]

        # l has list of points
        print("right_clicks: " + str(np.array(self._right_clicks)))
        print("correspondences: " + str(np.array(l)))
        H, status = cv2.findHomography(np.array(self._right_clicks), np.array(l))
        print H
        print status
        cv2.destroyAllWindows()
        self._img = cv2.warpPerspective(self._img,H, (self._img.shape[1],self._img.shape[0]))
        cv2.imwrite("output.png", self._img)
        cv2.imshow('image', self._img)
        cv2.waitKey(0)




##if __name__ == '__main__':
##    img = cv2.imread('00000.png',0)
##    getter = getHomography(img)
