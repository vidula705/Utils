
# python3 objDetect.py

# import the necessary packages
import numpy as np
import cv2
import sys
import datetime

conf = 0.5
thresh  = 0.3
DumpVideo = 0
showVideo = 1

 
gun_cascade = cv2.CascadeClassifier('cascade.xml')
# initialize the first frame in the video stream
firstFrame = None


"""--------------------------------------------------------------------------------------------
Function to write video to file
--------------------------------------------------------------------------------------------"""
def VideoWriter(writer, frame):
    # check if the video writer is None
    if writer is None:
    	# initialize  video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(out, fourcc, 30, (frame.shape[1], frame.shape[0]), True)
	   # write the output frame to disk
        writer.write(frame)   




"""--------------------------------------------------------------------------------------------
Function to file path
--------------------------------------------------------------------------------------------"""
def FilePaths():
    # load the COCO class labels on which pre YOLO model was trained on
    labelPath = "./weightFiles/coco.names" 
    Labels = open(labelPath).read().strip().split("\n")
    
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(Labels), 3),
    	dtype="uint8")
    
    # Path to the YOLO weights and model configuration files
    weightsPath = "./weightFiles/yolov3.weights"
    configPath  = "./weightFiles/yolov3.cfg"
    
    return Labels, colors, weightsPath, configPath


"""--------------------------------------------------------------------------------------------
Function to load model
--------------------------------------------------------------------------------------------"""
def LoadModel(configPath, weightsPath):
    # load the YOLO object detector pre-trained on COCO dataset (80 classes)
    print("Loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, ln
    

"""--------------------------------------------------------------------------------------------
Function to call main function
--------------------------------------------------------------------------------------------"""
def main():
    Labels, colors, weightsPath, configPath = FilePaths()
    net, ln = LoadModel(configPath, weightsPath)
    if DumpVideo:
       writer = None
    cap = cv2.VideoCapture(0)
    cap.set(3, 3264)
    cap.set(4, 2448)
    # loop over frames from the video file stream
    while cap.isOpened():
        # read the next frame from the file
        (grabbed, frame) = cap.read()
        frame = cv2.resize(frame, (640,480))
        if not grabbed:
            break
    
        (H, W) = frame.shape[:2]
    
    	# construct a blob from the input frame and then perform a forward
    	# pass of the YOLO object detector, giving the bounding boxes
    	# and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        #start = time.time()
        layerOutputs = net.forward(ln)
        #end = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        gun = gun_cascade.detectMultiScale(gray, 1.3, 5, minSize = (100, 100), maxSize = (300, 300))
        if len(gun) > 0:
            gun_exist = True  
            
        #for (x,y,w,h) in gun:
         #   frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
          #  cv2.putText(frame, "Gun", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            # draw the text and timestamp on the frame
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                       
    	# initialize our lists of detected bounding boxes, confidences,
    	# and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []
    
    	# loop over each of the layer outputs
        for output in layerOutputs:
    		# loop over each of the detections
            for detection in output:
    			# extract the class ID and confidence (i.e., probability)
    			# of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
    
    			# filter out weak predictions by ensuring the detected
    			# probability is greater than the minimum probability
                if confidence > conf:
    				# scale the bounding box coordinates back relative to
    				# the size of the image, keeping in mind that YOLO
    				# actually returns the center (x, y)-coordinates of
    				# the bounding box followed by the boxes' width and
    				# height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
    
    				# use the center (x, y)-coordinates to derive the top
    				# and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
    
    				# update our list of bounding box coordinates,
    				# confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
    
    	# apply non-maxima suppression to suppress weak, overlapping
    	# bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf, thresh)
    
    	# ensure at least one detection exists
        if len(idxs) > 0:
    		# loop over the indexes we are keeping
            for i in idxs.flatten():
    			# extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
    
    			# draw a bounding box rectangle and label on the frame
                color = [int(c) for c in colors[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}".format(Labels[classIDs[i]])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if DumpVideo:
            VideoWriter(writer, frame)
        if showVideo:
            cv2.imshow("Out",frame)
            key =  cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                print ('Close All windows ...')
                cap.release()   
                cv2.destroyAllWindows() 
                break
         
    print("Process completed...")
    
"""--------------------------------------------------------------------------------------------
Function to call main function
--------------------------------------------------------------------------------------------"""
if __name__ == '__main__':
     main()
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
