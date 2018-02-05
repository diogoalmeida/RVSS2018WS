import numpy as np
import PiBot as PB

CALIB = 0.360 / ((622 + 653)/2)
BASELINE = 12.499/90

class Odometry:
    def __init__(self, ip):
        self.bot = PB.PiBot(ip)
        self.reset()

    def getPos(self):
        ticks_now = np.array(self.bot.getMotorTicks())
        return (ticks_now - self.starting_ticks) * CALIB

    def getDistTheta(self):
        pos = self.getPos()
        dist = (pos[0] + pos[1]) / 2
        theta = (pos[0] - pos[1]) / BASELINE * 180/np.pi
        return dist, theta

    def getDistanceMeters(self):
        pos = self.getPos()
        dist = (pos[0] + pos[1]) / 2
        return dist

    def getThetaDegrees(self):
        pos = self.getPos()
        theta = (pos[0] - pos[1]) / BASELINE * 180/np.pi
        return theta

    def reset(self):
        self.starting_ticks = np.array(self.bot.getMotorTicks())

