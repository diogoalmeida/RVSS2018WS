#!usr/bin/python
import time
import math
from odometry import *
import threading
import cv2

DIST_THRESH  = 0.1  # 3 cm
THETA_THRESH = 20     # 3 degrees
directory = 'data_log_2'

class DataLogger:
    def __init__(self, ip):
        self.bot = PB.PiBot(ip)
        self.odo = Odometry(ip)
        self.count = 0

        self.file = open(directory + '/log.csv', 'w')

        self.running = True
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def run(self):
        self.record()
        while self.running:
            tic = time.time()
            meters, degrees = self.odo.getDistTheta()
            #print 'Took %5.3f sec to get odometry' % (toc - tic)
            print 'Dist: %5.3f meter, Theta: %5.3f deg' % (meters, degrees)

            if abs(meters) >= DIST_THRESH or abs(degrees) > THETA_THRESH:
                self.record()

            #time.sleep(0.001)
            toc = time.time()

            #print 'Loop took %5.3f sec' % (toc - tic)

    def record(self):
        self.bot.stop()

        meters, degrees = self.odo.getDistTheta()
        im = self.bot.getImageFromCamera()

        name = 'image_%06d.png' % (self.count)
        cv2.imwrite(directory + '/' + name, im)

        line = '%s %f %f\n' % (name, meters, degrees)
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
