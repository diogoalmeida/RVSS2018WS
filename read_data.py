import cv2
import getHomography as homography

filepath = "data_log.txt"
datapath = "test_log/"

count = 0
with open(datapath+filepath) as f:
    content = f.readlines()

    for line in content:
        if(count < 1):
	    data_text = line.rstrip('\n').split(" ")
	    img1 = cv2.imread(datapath+data_text[0])
	    homography.getHomography(img1)
        count = count + 1


