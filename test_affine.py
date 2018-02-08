# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 23:01:59 2018

@author: agniv
"""

import numpy as np
import cv2

H = np.array([[  1.12375325e-01,  -4.86001720e-01,   8.77712777e+02],
              [  -2.36834933e+00,   4.49234588e+00,   6.64034708e+02],
              [   0,   0,   1.00000000e+00]],float)
              
              
input_pts = np.array([[10,20],[11,21],[12,22],[13,23]])


op_pts = cv2.warpAffine( input_pts, H, (4,2) );

print H, input_pts
              


              
                  
                  
                  