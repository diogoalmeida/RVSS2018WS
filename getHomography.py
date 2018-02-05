import cv2
import numpy as np
import urllib


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

            if len(self._right_clicks) >= 4:
                self.compute_homography()

    def compute_homography(self):
        """Get points values and compute homography."""
        l = []
        for i in range(1,5):
            s = raw_input('insert p' + str(i) + ': ')
            p = list(map(float, s.split()))
            l = l + [p]

        # l has list of points
        H = cv2.findHomography(np.array(self._right_clicks), np.array(l))[0]
        cv2.destroyAllWindows()
        self._img =cv2.warpPerspective(self._img,H, (300,300))
        cv2.imshow('image', self._img)
        cv2.waitKey(0)


        print H


##if __name__ == '__main__':
##    img = cv2.imread('00000.png',0)
##    getter = getHomography(img)
