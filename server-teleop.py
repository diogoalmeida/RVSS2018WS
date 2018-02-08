#!usr/bin/python
import time
import math
import cv2
import numpy as np
#import sys
import pygame

import PiBot as PB

# write your command to initialise robot here
ip = '10.0.0.23'
bot = PB.PiBot(ip)

pygame.init()
pygame.display.set_mode((10,10))

speed = 50

try:
    print("TELEOP")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                print event.type
                if event.key == pygame.K_UP:
                    print("FORWARD")
                    bot.setMotorSpeeds(-speed, speed)
                if event.key == pygame.K_DOWN:
                    print("BACKWARD")
                    bot.setMotorSpeeds(speed, -speed)
                if event.key == pygame.K_LEFT:
                    print "LEFT"
                    bot.setMotorSpeeds(speed, speed)
                if event.key == pygame.K_RIGHT:
                    print "RIGHT"
                    bot.setMotorSpeeds(-speed, -speed)

            if event.type == pygame.KEYUP:
                print "Done"
                # stop the robot on Key up
                bot.stop()

except KeyboardInterrupt:
    bot.stop()
