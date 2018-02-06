import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

filepath = "data_log.txt"
datapath = "test_log/"



def estimate_motion(img1, img2, frame, filename):
    H = np.array([[  1.12375325e-01,  -4.86001720e-01,   8.77712777e+02],
                  [  -2.36834933e+00,   4.49234588e+00,   6.64034708e+02],
                  [   4.01039286e-04,   1.59276405e-02,   1.00000000e+00]],float)

    img1 = cv2.warpPerspective(img1, H, (600,600))
    img2 = cv2.warpPerspective(img2, H, (600,600))

    corners1 = cv2.goodFeaturesToTrack(img1, maxCorners=15, qualityLevel=0.1, minDistance=10) 
    corners1 = np.float32(corners1) 

    corners2 = cv2.goodFeaturesToTrack(img2, maxCorners=15, qualityLevel=0.1, minDistance=10) 
    corners2 = np.float32(corners2) 

    for item in corners1: 
        x, y = item[0] 
        cv2.circle(img1, (x,y), 5, 255, -1) 


    for item in corners2: 
        x, y = item[0] 
        cv2.circle(img2, (x,y), 5, 255, -1) 

    lk_params = dict( winSize  = (15,15),maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    p1, st, err = cv2.calcOpticalFlowPyrLK(img1,img2,corners1,None,**lk_params)

    good_new = p1[st==1]
    good_old = corners1[st==1]

    mask = np.zeros_like(frame)

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        #print a,b,c,d
        mask = cv2.line(mask, (a,b),(c,d), np.array([255,255,0]), 2)
        frame = cv2.circle(frame,(a,b),5,np.array([255,0,0]),-1)
    
    img_disp = cv2.add(frame,mask)

    trnsfrm = cv2.estimateRigidTransform(good_old, good_new, False)
    print trnsfrm
    print "Angle: "+str(np.clip(trnsfrm[0][0],0,1)) 
    angle = math.acos(np.clip(trnsfrm[0][0],0,1))*180/math.pi
    dist = math.sqrt((trnsfrm[0][2]*trnsfrm[0][2]) + (trnsfrm[1][2]*trnsfrm[1][2]))

    print filename,angle,dist

    with open("test_log/optical_estimate.txt", "a") as myfile:
        myfile.write(filename+","+str(angle)+","+str(dist)+"\n")


    numpy_horizontal = np.hstack((img1,img2))
    cv2.imshow('frame',img_disp)
    cv2.imshow('Numpy Horizontal', numpy_horizontal)
    cv2.waitKey()





prev_img = cv2.imread("/home/agniv/code/RVSS/RVSS_team/RVSS2018WS/test_log/00000.png",0)
prev_file_name = ""
count = 0
with open(datapath+filepath) as f:
    content = f.readlines()

    for line in content:
	data_text = line.rstrip('\n').split(" ")
        #print "reading: "+data_text
	img1 = cv2.imread(datapath+data_text[0],0)
        if(count>1):
            img2_color = cv2.imread(datapath+data_text[0])        
            img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
            estimate_motion(img1,img2,img2_color,prev_file_name)
	prev_img = img1
        prev_file_name = data_text[0]
        count = count + 1


