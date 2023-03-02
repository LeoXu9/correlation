import cv2
import numpy as np
from sklearn.cluster import KMeans
import PIL
import matplotlib

img1=cv2.imread('photos/Rossi.JPG',1)

hsv = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)

#lower red
lower_red = np.array([0,50,50])
upper_red = np.array([10,255,255])


#upper red
lower_red2 = np.array([170,50,50])
upper_red2 = np.array([180,255,255])

mask = cv2.inRange(hsv, lower_red, upper_red)
res = cv2.bitwise_and(img1,img1, mask= mask)


mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
res2 = cv2.bitwise_and(img1,img1, mask= mask2)

img3 = res+res2
# img4 = cv2.add(res,res2)
# img5 = cv2.addWeighted(res,0.5,res2,0.5,0)


kernel = np.ones((15,15),np.float32)/225

smoothed = cv2.filter2D(img3,-1,kernel)



# cv2.namedWindow('smooth2', cv2.WINDOW_NORMAL)
# cv2.imshow('smooth2',smoothed2)
# cv2.waitKey(0)
cv2.imwrite('res.jpg', smoothed)
im = PIL.Image.open("res.JPG")

black=0
red=0
for pixel in im.getdata():
    if pixel == (0,0,0):
        black += 1
    else:
        red += 1
print(red)