# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:04:19 2018

@author: vidula
"""
'''--------------------------------------------------------------------------------------------'''
'''Python Implementation for KITTI data visualization '''
'''--------------------------------------------------------------------------------------------'''

'''Include library'''
import numpy as np
import cv2
import os

'''Input Paths'''
imagePathTxt  = r'F:\DL\BigData\kitti\ImageData.txt'.replace('\\','/') #Text file listing Image details
baseImageDir  = r'F:\DL\BigData\kitti\image/'.replace('\\','/')   #Image base folder

"""--------------------------------------------------------------------------------------------
Function to read the frames from the video and call for process frame function
--------------------------------------------------------------------------------------------"""
def DumpImageSize(ImageList):
    file = open('ImageSize_ratio.txt','a')
    for ImageNam in (ImageList):
        ImageName =  ((ImageNam.decode("utf-8") ))
        ImagePath   = baseImageDir + ImageName
        if not os.path.isfile(ImagePath):
            print ('FILE MISSING: ' + ImagePath)
            break
        print ("Image_Name:  ",ImageName)
        Ipl_Image = cv2.imread(ImagePath) 
        print ((Ipl_Image.shape))
        aspect_ratio = Ipl_Image.shape[1] / Ipl_Image.shape[0]
        file.write("%s ( %d , %d )  %f\n" % (ImageName, Ipl_Image.shape[1],Ipl_Image.shape[0],aspect_ratio ))
    file.close()
        

"""--------------------------------------------------------------------------------------------
Function to read the Image details from the text file
--------------------------------------------------------------------------------------------"""
def ReadImageFile():
    ImagefileList = np.genfromtxt(imagePathTxt, dtype = np.object)
    DumpImageSize(ImagefileList )
 

"""--------------------------------------------------------------------------------------------
Function to dispaly frames
--------------------------------------------------------------------------------------------"""
def DisplayImage(WindowName, OutputImage):
    cv2.namedWindow(WindowName, cv2.WINDOW_NORMAL)
    cv2.imshow(WindowName, OutputImage)      
        
"""--------------------------------------------------------------------------------------------
Function to save frames
--------------------------------------------------------------------------------------------"""
def SaveImage(ImageName, OutputImage):
    dumpPath = saveDir
    if not os.path.exists(dumpPath):
        os.makedirs(dumpPath)
    cv2.imwrite(dumpPath+'/'+ImageName, OutputImage)
    
    
"""--------------------------------------------------------------------------------------------
Main function
--------------------------------------------------------------------------------------------"""   
if __name__ == "__main__":
    
    ReadImageFile()
    
    cv2.destroyAllWindows()

'''--------------------------------------------------------------------------------------------'''
'''End of the File'''
'''--------------------------------------------------------------------------------------------'''

