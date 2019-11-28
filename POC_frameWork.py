# -*- coding: utf-8 -*-
"""
Created on Thu Oct 05 14:43:36 2017

@author: vidula
"""
'''--------------------------------------------------------------------------------------------'''
'''Basic Python Framework for POC Development '''
'''--------------------------------------------------------------------------------------------'''

'''Include library'''
import numpy as np
import cv2
import os

'''Input Paths'''
imagePathTxt = r'.\VideoData.txt'.replace('\\','/') #Text file listing video details
baseImageDir = r'K:\TLR_SLD\video_data\09-09_JARI_sonySensor/'.replace('\\','/')   #Video base folder
saveDir      = r'.\DebugResults/'.replace('\\','/')                            #Dump directory

'''Macros'''
ENABLE_BATCH_RUN    = 1
ENABLE_DEBUG_DUMP   = 0
FRAME_STEP          = 2

if not ENABLE_BATCH_RUN:
    '''To process single video'''
    VideoName       = '0002'     #Video name
    VideoPrefix     = '0001_'    #Video Prefix
    StartFrameName  = 11305      #Video start frame name
    EndFrameName    = 11385      #Video end frame name

"""--------------------------------------------------------------------------------------------
Function to read the frames from the video and call for process frame function
--------------------------------------------------------------------------------------------"""
def ReadVideo(VideoName, VideoPrefix, StartFrameName, EndFrameName):
    VideoPath   = baseImageDir + VideoName
    for x in range(StartFrameName, EndFrameName, FRAME_STEP):
       ImageName = VideoPrefix + "%06d.bmp" %(x)
       ImgPath   = VideoPath + '/' + ImageName
       if not os.path.isfile(ImgPath):
           print 'FILE MISSING: ' + ImgPath
           break
       Ipl_Image = cv2.imread(ImgPath)
       print ImageName
      #Call to core modules
       OutputImage = ProcessFrame(Ipl_Image)

       Debug(VideoName, ImageName, OutputImage)

       k = cv2.waitKey(1) & 0xFF
       if (k == ord('q')) or (k == ord('Q')):
          break

"""--------------------------------------------------------------------------------------------
Function to read the video details from the text file
--------------------------------------------------------------------------------------------"""
def ReadBatch():
    fileList = np.genfromtxt(imagePathTxt, dtype = np.object)
    for data in range(0, fileList.shape[0]):
        ReadVideo(fileList[data][0], fileList[data][1], int(fileList[data][2]), int(fileList[data][3]))
        k = cv2.waitKey(1) & 0xFF
        if (k == ord('q')) or (k == ord('Q')):
            break

"""--------------------------------------------------------------------------------------------
Function to call core modules
--------------------------------------------------------------------------------------------"""
def ProcessFrame(InputImage):
    print "Function Call"
    OutputImage = (InputImage)

    return OutputImage

"""--------------------------------------------------------------------------------------------
Function to save frames
--------------------------------------------------------------------------------------------"""
def SaveImage(VideoName, ImageName, OutputImage):
    dumpPath = saveDir+VideoName
    if not os.path.exists(dumpPath):
        os.makedirs(dumpPath)
    cv2.imwrite(dumpPath+'/'+ImageName, OutputImage)

"""--------------------------------------------------------------------------------------------
Function to dispaly frames
--------------------------------------------------------------------------------------------"""
def DisplayImage(WindowName, OutputImage):
    cv2.namedWindow(WindowName, cv2.WINDOW_NORMAL)
    cv2.imshow(WindowName, OutputImage)

"""--------------------------------------------------------------------------------------------
Function to Debug frames
--------------------------------------------------------------------------------------------"""
def Debug(VideoName, ImageName, OutputImage):
    if ENABLE_DEBUG_DUMP:
        SaveImage(VideoName, ImageName, OutputImage)
    DisplayImage('DisplayWindow', OutputImage)

"""--------------------------------------------------------------------------------------------
Main function
--------------------------------------------------------------------------------------------"""
if __name__ == "__main__":
    if ENABLE_BATCH_RUN:
       ReadBatch()
    else:
        ReadVideo(VideoName, VideoPrefix, StartFrameName, EndFrameName)

    cv2.destroyAllWindows()

'''--------------------------------------------------------------------------------------------'''
'''End of the File'''
'''--------------------------------------------------------------------------------------------'''






