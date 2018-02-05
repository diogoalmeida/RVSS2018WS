#!usr/bin/python
import time
import math
from odometry import *
import threading
import cv2

DIST_THRESH  = 0.03  # 3 cm
THETA_THRESH = 3     # 3 degrees

class DataLogger:
    def __init__(self, bot):
        self.bot = bot
        self.odo = Odometry(bot)
        self.count = 0

        self.file = open('data_log/log.csv', 'w')

        self.running = True
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def run(self):
        while self.running:
            tic = time.time()
            meters, degrees = self.odo.getDistTheta()
            toc = time.time()
            print 'Took %5.3f sec to get odometry' % (toc - tic)
            #print 'Dist: %5.3f meter, Theta: %5.3f deg' % (meters, degrees)

            if meters >= DIST_THRESH or degrees > THETA_THRESH:
                print 'Record'
                self.record()

            time.sleep(0.1)

    def record(self):
        self.bot.stop()
        tic = time.time()
        im = self.bot.getImageFromCamera()
        toc = time.time()
        print 'Took %5.3f sec to get image' % (toc - tic)

        cv2.imwrite('data_log/image_%06d.png' % (self.count), im)

        line = '%d, %f, %f\n' % (self.count, self.odo.getDistanceMeters(), self.odo.getThetaDegrees())
        self.file.write(line)
        print line

        self.count += 1
        self.odo.reset()

    def stop(self):
        self.running = False

        # wait for any pending events
        time.sleep(0.1)
        self.file.close()
