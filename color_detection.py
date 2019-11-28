import cv2 
import numpy as np  
  
# Webcamera no 0 is used to capture the frames 
#cap = cv2.VideoCapture(0)  
window_name='color range parameter'
def nothing(x):
    pass
cv2.namedWindow(window_name)
# Create a black image, a window
im = cv2.imread('/home/vidula/BOLT/VEGETABLE_DETECTION_BASED_ON_COLOR/imgs/3.jpg')
cb = cv2.imread('/home/vidula/BOLT/VEGETABLE_DETECTION_BASED_ON_COLOR/imgs/3.jpg')
hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)

print ('lower_color = np.array([a1,a2,a3])')
print ('upper_color = np.array([b1,b2,b3])')


# create trackbars for color change
cv2.createTrackbar('a1',window_name,0,255,nothing)
cv2.createTrackbar('a2',window_name,0,255,nothing)
cv2.createTrackbar('a3',window_name,0,255,nothing)

cv2.createTrackbar('b1',window_name,150,255,nothing)
cv2.createTrackbar('b2',window_name,150,255,nothing)
cv2.createTrackbar('b3',window_name,150,255,nothing)

 
# This drives the program into an infinite loop. 
while(1):        
	# Captures the live stream frame-by-frame 
	#_, frame = cap.read()  
	# Converts images from BGR to HSV 
	#hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 

	a1 = cv2.getTrackbarPos('a1',window_name)
	a2 = cv2.getTrackbarPos('a2',window_name)
	a3 = cv2.getTrackbarPos('a3',window_name)

	b1 = cv2.getTrackbarPos('b1',window_name)
	b2 = cv2.getTrackbarPos('b2',window_name)
	b3 = cv2.getTrackbarPos('b3',window_name)


	lower_color = np.array([a1,a2,a3])
	upper_color = np.array([b1,b2,b3])
	mask = cv2.inRange(hsv, lower_color, upper_color)
	res = cv2.bitwise_and(im, im, mask = mask)

	cv2.imshow('mask',mask)
	cv2.imshow('res',res)
	cv2.imshow('im',im)
	cv2.imshow(window_name,cb)

	k = cv2.waitKey(1) & 0xFF
	if k == 27:         # wait for ESC key to exit
		break
	elif k == ord('s'): # wait for 's' key to save and exit
		cv2.imwrite('Img_screen_mask.jpg',mask)
		cv2.imwrite('Img_screen_res.jpg',res)
		break
  
# Destroys all of the HighGUI windows. 
cv2.destroyAllWindows() 
  
# release the captured frame 
#cap.release() 

