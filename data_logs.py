#!usr/bin/python
import time
import math
from odometry import *
import threading
import cv2

DIST_THRESH  = 0.03  # 3 cm
THETA_THRESH = 3     # 3 degrees

class DataLogger:
    def __init__(self, ip):
        self.bot = PB.PiBot(ip)
        self.odo = Odometry(ip)
        self.count = 0

        self.file = open('data_log/log.csv', 'w')

        self.running = True
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def run(self):
        while self.running:
            tic = time.time()
            meters, degrees = self.odo.getDistTheta()
            #print 'Took %5.3f sec to get odometry' % (toc - tic)
            #print 'Dist: %5.3f meter, Theta: %5.3f deg' % (meters, degrees)

            if abs(meters) >= DIST_THRESH or abs(degrees) > THETA_THRESH:
                self.record()

            #time.sleep(0.001)
            toc = time.time()

            #print 'Loop took %5.3f sec' % (toc - tic)

    def record(self):
        self.bot.stop()

        meters, degrees = self.odo.getDistTheta()
        im = self.bot.getImageFromCamera()

        cv2.imwrite('data_log/image_%06d.png' % (self.count), im)

        line = '%d, %f, %f\n' % (self.count, meters, degrees)
        self.file.write(line)
        print line

        self.count += 1
        self.odo.reset()

    def stop(self):
        self.running = False

        # wait for any pending events
        time.sleep(0.1)
        self.file.close()

print 'Data Logger'

ip = '10.0.0.23'
logger = DataLogger(ip)

print 'Press Enter to stop'
raw_input()

logger.stop()
print 'Stopped'


