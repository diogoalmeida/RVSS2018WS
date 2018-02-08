import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

filepath = "data_log.txt"
datapath = "test_log/"



def estimate_motion(img1, img2, frame, filename, x, y,distX,distY,sum_dist):
    H = np.array([[  1.12375325e-01,  -4.86001720e-01,   8.77712777e+02],
                  [  -2.36834933e+00,   4.49234588e+00,   6.64034708e+02],
                  [   4.01039286e-04,   1.59276405e-02,   1.00000000e+00]],float)

    #img1 = cv2.warpPerspective(img1, H, (600,600))
    #img2 = cv2.warpPerspective(img2, H, (600,600))

    corners1 = cv2.goodFeaturesToTrack(img1, maxCorners=1000, qualityLevel=0.1, minDistance=2) 
    corners1 = np.float32(corners1) 

    corners2 = cv2.goodFeaturesToTrack(img2, maxCorners=1000, qualityLevel=0.1, minDistance=2) 
    corners2 = np.float32(corners2) 

    for item in corners1: 
        x, y = item[0] 
        cv2.circle(img1, (x,y), 5, 255, -1) 


    for item in corners2: 
        x, y = item[0] 
        cv2.circle(img2, (x,y), 5, 255, -1) 

    lk_params = dict( winSize  = (11,11),maxLevel = 5, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 0.03))

    p1, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, corners1, None, **lk_params)
    
    

    good_new = p1[st==1]
    good_old = corners1[st==1]
    
    if len(good_new) > 0 and len(good_old) > 0:
    
        #good_new = H*good_new
       
        #print good_new
        #print good_new.transpose()
        
        good_new = np.array([good_new])
        good_old = np.array([good_old])
    
        print good_new
        print good_old
        good_new = cv2.perspectiveTransform(good_new,H)[0]
        good_old = cv2.perspectiveTransform(good_old,H)[0]
        
    
        #print good_new.shape
        
        frame = cv2.warpPerspective(frame,H,(600,600))
        mask = np.zeros_like(frame)
    
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            #print a,b,c,d
            mask = cv2.line(mask, (a,b),(c,d), np.array([255,255,0]), 2)
            frame = cv2.circle(frame,(a,b),5,np.array([0,0,0]),-1)
            frame = cv2.circle(frame,(c,d),5,np.array([255,255,255]),-1)
        
        img_disp = cv2.add(frame,mask)
        
        #print good_old
        #print good_new
    
        trnsfrm = cv2.estimateRigidTransform(good_old, good_new, False)
        #trnsfrm = cv2.getAffineTransform(good_old, good_new)
        
        #print trnsfrm
        
        
        if trnsfrm is None:
            trnsfrm = np.array([[1, 0, 0], [0, 1, 0]])
            
        angle = math.atan2(trnsfrm[1][0],trnsfrm[1][1])*180/math.pi
        dist = math.sqrt((trnsfrm[0][2]*trnsfrm[0][2]) + (trnsfrm[1][2]*trnsfrm[1][2]))
        
        sum_dist = sum_dist + dist
        
        with open("test_log/optical_estimate.txt", "a") as myfile:
                myfile.write(filename+" "+str(angle)+" "+str(dist)+"\n")
        
        vis = np.array([distX, distY,1])
        
        #print distX, distY
        a = np.array([[0,0,1]])
    
        trnsfrm = np.concatenate((trnsfrm, a), axis=0)
        
        print trnsfrm
        
        
        vis = np.dot(trnsfrm,vis)
        print vis
        #print angle, dist
        
        
        numpy_horizontal = np.hstack((img1,img2))
        cv2.imshow('frame',img_disp)
        cv2.imshow('Numpy Horizontal', numpy_horizontal)
        cv2.waitKey()
        
        
        '''
        
        if trnsfrm is not None:
            #print "Transformation matrix:"
            #print trnsfrm
            #print "Angle from matrix: "+str(trnsfrm[0][0])
            #print "Angle: "+str(np.clip(trnsfrm[0][0],-1,1)) 
            #angle_x = np.clip(trnsfrm[0][0],-1,1)
            angle = math.atan2(trnsfrm[1][0],trnsfrm[1][1])*180/math.pi
            dist = math.sqrt((trnsfrm[0][2]*trnsfrm[0][2]) + (trnsfrm[1][2]*trnsfrm[1][2]))
            
            print angle, dist
            
            
            #print "dist, dist*c, dist*s, tx, ty, c, s"        
            #print dist, (dist*np.clip(trnsfrm[0][0],-1,1)), (dist*np.clip(trnsfrm[1][0],-1,1)), trnsfrm[0][2],trnsfrm[1][2],np.clip(trnsfrm[0][0],-1,1),np.clip(trnsfrm[1][0],-1,1)
        
            #print filename,angle,dist
        
            with open("test_log/optical_estimate.txt", "a") as myfile:
                myfile.write(filename+","+str(angle)+","+str(dist)+"\n")
        
        
            numpy_horizontal = np.hstack((img1,img2))
            cv2.imshow('frame',img_disp)
            cv2.imshow('Numpy Horizontal', numpy_horizontal)
            cv2.waitKey()
        
            #print "pose update: "+str(math.cos(angle))+" angle: "+str(angle)
            #print "pose update: "+str(math.sin(angle))+" angle: "+str(angle)
        
            distX = distX + (dist*np.clip(trnsfrm[0][0],-1,1))#math.cos(angle)
            distY = distY + (dist*np.clip(trnsfrm[1][0],-1,1))#math.sin(angle)
            print distX, distY
    
            if angle > 0:
                distY = distY + math.sin(angle)
            elif angle <= 0:
                distY = distY + math.sin(angle)
            return distX, distY
            
        else:
            return 0,0
            
        '''
    else:
        vis = np.array([0, 0,1])
    
    return vis[0], vis[1], sum_dist

    





prev_img = cv2.imread("/home/agniv/code/RVSS/RVSS_team/RVSS2018WS/test_log/00000.png",0)
prev_file_name = ""
count = 0

x = np.zeros((1), int)
y = np.zeros((1), int)

x = np.append(x, np.array([0]), axis=0)
y = np.append(y, np.array([0]), axis=0)

distX = 0
distY = 0

sum_dist = 0

with open(datapath+filepath) as f:
    content = f.readlines()
    for line in content:
	data_text = line.rstrip('\n').split(" ")
        #print "reading: "+data_text
	img1 = cv2.imread(datapath+data_text[0],0)
        if(count>1):
            img2_color = cv2.imread(datapath+data_text[0])        
            img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
            distX, distY, sum_dist = estimate_motion(prev_img,img2,img2,prev_file_name,x,y,distX,distY,sum_dist)
            x = np.append(x, np.array([distX]), axis=0)
            y = np.append(y, np.array([distY]), axis=0)
            prev_img = img1
            prev_file_name = data_text[0]
        count = count + 1



print sum_dist

print x.shape
print y.shape
plt.scatter(x, y)
#plt.axis([-50, 50, -50, 50])
plt.show()