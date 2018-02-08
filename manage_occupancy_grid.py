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
        self._detection_thres = .4

        self.total_img = None
        self.count = None

    def show_grid(self):
        cv2.imshow("map", self._grid)
        cv2.waitKey(0)

    def update_map_pose(self, fn, pose):
        pose_img = [0, 0]
        pose_img[0] = int(pose[0]/self._res + self._dim[0]/2)
        pose_img[1] = int(pose[1]/self._res + self._dim[1]/2)

        scale_factor = 0.1 / self._res

        rot_y = 300
        M = cv2.getRotationMatrix2D((0, rot_y), np.rad2deg(pose[2]), scale_factor) # includes scale from cm to mm

        M[0, 2] = M[0,2] + pose_img[0]
        M[1, 2] = M[1,2] + pose_img[1] - rot_y

        # print M
        print fn
        img = cv2.imread(img_dir + '/' + fn)
        mask = img * 0 + 1

        reshape = (img.shape[1], img.shape[0])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.warpPerspective(img, H, self._dim, borderValue = -1) # img in the robot frame
        #img = cv2.resize(img, reshape, interpolation = cv2.INTER_AREA)
        img = cv2.warpAffine(img, M, self._dim, borderValue = -1) # img in the world frame

        reshape = (mask.shape[1], mask.shape[0])
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        mask = cv2.warpPerspective(mask, H, self._dim) # img in the robot frame
        #mask = cv2.resize(mask, reshape, interpolation = cv2.INTER_CUBIC)
        mask = cv2.warpAffine(mask, M, self._dim) # img in the world frame
        mask_xy = np.argwhere(mask > 0)

        # cv2.imshow("img", img)
        # cv2.waitKey(0)

        tmp = np.zeros(self._dim, float)
        # img should be downsampled in a way we can add squares to the grid

        for x, y in mask_xy:
            g_x = x
            g_y = y
            if 0 <= g_x and g_x < self._dim[0] and 0 <= g_y and g_y < self._dim[1]:
                tmp[g_x][g_y] = img[x][y]/255.0 #self._grid[g_x][g_y] + 0.1
                #tmp[g_x][g_y] = mask[x][y]

        if self.total_img is None:
            self.total_img = tmp
            self.count = mask
        else:
            self.total_img += tmp
            self.count += mask

        render = self.total_img / (self.count + 0.00001)
        cv2.circle(render, (pose_img[0], pose_img[1]), 4, color=(235,20,20), thickness=-1)

        cv2.imshow("warped img", render)
        cv2.waitKey(1)
        #time.sleep(0.01)

    def update_map_slam(self, est):
        for i, e  in enumerate(est):
            print i, e
            e[1] *= -1
            self.update_map_pose('%05d.png' % i, e)

        render = self.total_img / (self.count + 0.00001)
        mask_improv = np.argwhere(render < self._detection_thres)
        img = np.zeros(self._dim)
        for x, y in mask_improv:
                g_x = x
                g_y = y
                if 0 <= g_x and g_x < self._dim[0] and 0 <= g_y and g_y < self._dim[1]:
                    img[g_x][g_y] = 1

        img = 1 - img

        # img = self.total_img / (self.count + 1e-10)
        cv2.imshow("warped img", img)
        cv2.waitKey()
        print cv2.imwrite('out.png', img*255)

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

            self.update_map_pose(obs[0], self._pose)

        # cv2.imshow("warped img", self.total_img)
        # cv2.waitKey()
        render = self.total_img / (self.count + 0.00001)
        mask_improv = np.argwhere(render < self._detection_thres)
        img = np.zeros(self._dim)
        for x, y in mask_improv:
                g_x = x
                g_y = y
                if 0 <= g_x and g_x < self._dim[0] and 0 <= g_y and g_y < self._dim[1]:
                    img[g_x][g_y] = 1

        img = 1 - img

        # img = self.total_img / (self.count + 1e-10)
        cv2.imshow("warped img", img)
        cv2.waitKey()
        print cv2.imwrite('out.png', img*255)

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
    dim = (1000, 1000)
    resolution = .5 # cm per pixel

    observations = parse_obs(img_dir + "/data_log.txt", resolution)
    m = map(dim, resolution)
    m.update_map(observations)
    #m.show_grid()
    raw_input()
