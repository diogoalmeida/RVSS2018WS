from slam import *
import numpy as np
import matplotlib.pyplot as plt

class GraphOpt:
    def __init__(self, odo_dist, odo_theta, visual_dist, visual_theta):
        self.odo_dist = odo_dist
        self.odo_theta = odo_theta

        self.visual_dist = visual_dist
        self.visual_theta = visual_theta

        self.odo_global = self.convertToGlobal(odo_dist, odo_theta)
        self.visual_global = self.convertToGlobal(visual_dist, visual_theta)

        self.odo_relative = self.getRelative(self.odo_global)
        self.visual_relative = self.getRelative(self.visual_global)

        self.slam = Slam(self.odo_global, self.odo_relative, self.visual_global, self.visual_relative)
        estimated_global_trajectory = self.slam.estimate_global()

        self.est = np.array(estimated_global_trajectory)

    def convertToGlobal(self, dist, theta):
        xyt = np.zeros((dist.shape[0]+1, 3))

        for i in range(1, dist.shape[0]+1):
            d = dist[i-1]
            t = np.deg2rad(theta[i-1])

            theta_prev = xyt[i-1, 2]

            c = np.cos(theta_prev)
            s = np.sin(theta_prev)

            xyt[i, 0] = d * c
            xyt[i, 1] = d * s
            xyt[i, 2] = t

            xyt[i, :] += xyt[i-1, :]

        return xyt

    def getRelative(self, xyt):
        return np.diff(xyt, axis=0)

if __name__ == '__main__':
    def get_odo():
        data = np.loadtxt('data_log/log.csv', delimiter=',')
        dist = data[:, 1]
        theta = data[:, 2]
        return dist, theta

    def get_visual():
        dist, theta = get_odo()

        dist += np.random.normal(scale=0.02, size=dist.shape)
        theta += np.random.normal(scale=1.0, size=theta.shape)

        return dist, theta

    odo_dist, odo_theta = get_odo()
    vo_dist, vo_theta = get_visual()

    go = GraphOpt(odo_dist, odo_theta, vo_dist, vo_theta)

    plt.plot(go.odo_global[:, 0], go.odo_global[:, 1], 'bx-')
    plt.plot(go.visual_global[:, 0], go.visual_global[:, 1], 'rx-')
    plt.plot(go.est[:, 0], go.est[:, 1], 'kx-')
    plt.axis('equal')
    plt.grid()
    plt.show()
