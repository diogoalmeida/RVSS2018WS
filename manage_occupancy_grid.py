import numpy as np
import cv2
from IPython import embed
from matplotlib import pyplot as plt
import time

img_dir = './test_log'
H = np.array([[  1.12375325e-01,  -4.86001720e-01,   8.77712777e+02],
             [  -2.36834933e+00,   4.49234588e+00,   6.64034708e+02],
             [   4.01039286e-04,   1.59276405e-02,   1.00000000e+00]],float)

class map:
    def __init__(self, grid_dim, resolution):
        """Resolution: how many mm^2 per cell."""
        self._res = resolution
        self._dim = grid_dim
        self._thres = 125
        self._grid = np.zeros(grid_dim, float)
        self._pose = np.array([0., 0., 0.], float)
        self._rect_pose_y = 0.

        self.total_img = None
        self.count = None

    def show_grid(self):
        cv2.imshow("map", self._grid)
        cv2.waitKey(0)

    def update_map(self, observations):
        """Convert images to the correct map pose and decide
           whether a cell is occupied or not.

           @observations: tupple (img, delta_pose).
        """
        for obs in observations:
            # print obs
            self._pose[0] = self._pose[0] + obs[1]*np.cos(self._pose[2])
            self._rect_pose_y = self._rect_pose_y + obs[1]*np.sin(self._pose[2])
            self._pose[1] = self._pose[1] - obs[1]*np.sin(self._pose[2])
            self._pose[2] = self._pose[2] + obs[2]
            # print self._pose[2]
            plt.plot(self._pose[0], self._rect_pose_y, 'ro')
            plt.draw()

            pose_img = [0, 0]
            pose_img[0] = int(self._pose[0]/self._res + self._dim[0]/2)
            pose_img[1] = int(self._pose[1]/self._res + self._dim[1]/2)

            # M = self.get_M(pose_img + [self._pose[2]])
            scale_factor = 0.1 * self._res
            M = cv2.getRotationMatrix2D((0, 133), np.rad2deg(self._pose[2]), scale_factor) # includes scale from cm to mm

            M[0, 2] = M[0,2] + pose_img[0]
            M[1, 2] = M[1,2] + pose_img[1] - 133

            # print M

            img = cv2.imread(img_dir + '/' + obs[0])
            mask = img * 0 + 1

            reshape = (img.shape[1], img.shape[0])
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.warpPerspective(img, H, self._dim, borderValue = -1) # img in the robot frame
            img = cv2.resize(img, reshape, interpolation = cv2.INTER_AREA)
            img = cv2.warpAffine(img, M, self._dim, borderValue = -1) # img in the world frame

            reshape = (mask.shape[1], mask.shape[0])
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            mask = cv2.warpPerspective(mask, H, self._dim) # img in the robot frame
            mask = cv2.resize(mask, reshape, interpolation = cv2.INTER_AREA)
            mask = cv2.warpAffine(mask, M, self._dim) # img in the world frame
            mask_xy = np.argwhere(mask > 0)

            # cv2.imshow("img", img)
            # cv2.waitKey(0)

            tmp = np.zeros(self._dim, float)
            # img should be downsampled in a way we can add squares to the grid

            for x, y in mask_xy:
                #g_x = int(pose_img[1] + x - 133)
                #g_y = int(pose_img[0] + y)
                g_x = x
                g_y = y
                if 0 < g_x and g_x < self._dim[0] and 0 < g_y and g_y < self._dim[1]:
                    tmp[g_x][g_y] = img[x][y]/255.0 #self._grid[g_x][g_y] + 0.1

            if self.total_img is None:
                self.total_img = tmp
                self.count = mask
            else:
                self.total_img += tmp
                self.count += mask

            # temp = self._grid + resized_image
            # temp = cv2.addWeighted(self._grid, 0.5, resized_image,0.5,0)

            render = self.total_img / self.count
            cv2.circle(render, (pose_img[0], pose_img[1]), 4, (255,255,255), -1)

            cv2.imshow("warped img", render)
            cv2.waitKey(1)
            time.sleep(0.1)
        # cv2.waitKey(0)
        plt.show()

    def get_M(self, pose):
        x = 0*pose[0]
        y = 0*pose[1]
        angle = pose[2]

        s = np.sin(angle)
        c = np.cos(angle)
        M = np.array([[c, -s, x], [s, c, y]], float)

        return M

    def get_motion(self, delta_pos):
        """Get the matrix that shifts coordinates between two poses
           related by delta_pos = (delta_distance, delta_angle)."""

        # print delta_pos
        dist = delta_pos[0]
        angle = np.deg2rad(delta_pos[1])

        s = np.sin(angle)
        c = np.cos(angle)
        dx = dist*c
        dy = dist*s

        M = self.get_M(np.array([dx, dy, angle]))

        # adjust for the centre of rotation. TODO: compensate for camera movement?
        cr = np.array([self._pose[0] + self._res[0]/2, self._pose[1] + self._res[1]/2])
        R  = M[:2,:2]
        tc = R.dot(cr) - cr
        M[0,2] -= tc[0]
        M[1,2] -= tc[1]

        return M


def parse_obs(file_dir, res):
    data = open(file_dir)
    obs = []

    for line in data:
        l = line.split()
        l[1] = float(l[1])
        l[2] = np.deg2rad(float(l[2]))
        obs = obs + [l]

    return obs

if __name__ == "__main__":
    dim = (500, 500)
    resolution = 1 # mm^2 per cell

    observations = parse_obs(img_dir + "/data_log.txt", resolution)
    m = map(dim, resolution)
    m.update_map(observations)
    m.show_grid()
