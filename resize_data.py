# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 12:05:39 2018

@author: vidula
"""
import numpy as np
import glob
import cv2
import dataAug as aug
import pandas as pd
import os

#ImagePath = glob.glob("F:\\DL\\BigData\\kitti\image\\*.png")
#labeltxt = glob.glob("F:\\DL\\BigData\\kitti\\label\\*.txt")

ImagePath = glob.glob("F:\\DL\\BigData\\DataAug\\test\\*.png")
labeltxt  = glob.glob("F:\\DL\\BigData\\DataAug\\lab\\*.txt")

Image_width  = 1280
Imae_height  = 720
Imae_height1 = 392
Crop_percent = 0.10

font = cv2.FONT_HERSHEY_SIMPLEX

Debug = 0
Dump_AugData = 0
Dump_ResizedData = 0

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
    dumpPath = "./Resized_Data"
    if not os.path.exists(dumpPath):
        os.makedirs(dumpPath)
    cv2.imwrite(dumpPath+'/'+ImageName+'.jpeg', OutputImage)


"""--------------------------------------------------------------------------------------------
Resize Buunding Box Function 
--------------------------------------------------------------------------------------------"""
def resizeImgBB(InputImage, ResizedImage, Labeltxt, shape):
    bb_info = []
    for cand in range(0, shape[0]):
        heightRatio = round (ResizedImage.shape[0]/InputImage.shape[0], 2)
        widthRatio = round (ResizedImage.shape[1]/InputImage.shape[1], 2)
        bb_modX1 = round((float(Labeltxt[cand][4])*(widthRatio)), 2)
        bb_modY1 = round((float(Labeltxt[cand][5])*(heightRatio)), 2)
        bb_modX2 = round((float(Labeltxt[cand][6])*(widthRatio)), 2)
        bb_modY2 = round((float(Labeltxt[cand][7])*(heightRatio)), 2)
        bb_class = Labeltxt[cand][0]
        bb_info.append([bb_modX1, bb_modY1, bb_modX2, bb_modY2, bb_class])
    return bb_info

"""--------------------------------------------------------------------------------------------
Check Boundary box area Function 
--------------------------------------------------------------------------------------------"""

def Check_BBprecent(Ipl_Image,bb_info):
    valid = 1
    for cand in range(0, shape[0]):
        if (((float(bb_info[cand][2]) != 0) or (float(bb_info[cand][0]) != 0)) and (
        (float(bb_info[cand][3] != 0) or float(bb_info[cand][1]) != 0))):
            bbox_l = float(bb_info[cand][2]) - float(bb_info[cand][0])
            bbox_h = float(bb_info[cand][3]) - float(bb_info[cand][1])
            bbox_area = ((bbox_l) * (bbox_h))
            bb_percent = float(Ipl_Image.shape[1] * Ipl_Image.shape[0] * 0.65)
            if bbox_area > bb_percent:
                valid = 0
                print("Dicard Image")
                break
    return valid
 

"""--------------------------------------------------------------------------------------------
Calculate cropping width Function
--------------------------------------------------------------------------------------------"""   
def CropImage_Width(Ipl_Image, Labeltxt):
     Width_ratio  = round((Ipl_Image.shape[1] / Image_width), 2)
     CropRight    = int(Ipl_Image.shape[1]) - ((Ipl_Image.shape[1] / 2) * Crop_percent)
     CropLeft     = 0 + ((Ipl_Image.shape[1] / 2) * Crop_percent)

     df = pd.DataFrame(Labeltxt, columns=['X1','Y1','X2','Y2','Class'])
     
     BB_min_X1 = df['X1'].min()
     BB_max_X2 = df['X2'].max()
     index_min = df.loc[df['X1'].idxmin()]
     index_max = df.loc[df['X2'].idxmax()]
     if (Width_ratio > 0.33):
         if(BB_min_X1 < CropLeft) or (BB_max_X2 > CropRight):
             if (BB_min_X1 != 0 and BB_min_X1 <= CropLeft and index_min[2] < (Ipl_Image.shape[1]/2)):
                     bb_MidWL = (((index_min[2] - BB_min_X1 )/2) + BB_min_X1)
                     if(bb_MidWL <= CropLeft):
                         if(index_min[4] == 'Pedestrian'):
                             mod_LeftWidth = 0.0
                         elif(bb_MidWL == CropLeft):
                             mod_LeftWidth = int(CropLeft)
                             df = df.replace(BB_min_X1, CropLeft)
                         elif (bb_MidWL < CropLeft):
                             mod_LeftWidth = int(CropLeft) 
                             df = df.replace(BB_min_X1, 0.0)
                             df = df.replace(index_min[2], 0.0)
                         else:
                             mod_LeftWidth = 0.0
                     else:
                         df = df.replace(BB_min_X1, CropLeft)
                         mod_LeftWidth = int(CropLeft)
             else:
                mod_LeftWidth = 0.0 #check
                         
             if (BB_max_X2 != 0 and BB_max_X2 >= CropRight and index_max[0] > (Ipl_Image.shape[1]/2)):
                    bb_MidR = (((BB_max_X2 - index_max[0] )/2) + BB_min_X1)
                    if(bb_MidR >= CropRight):
                        if (index_max[4] == 'Pedestrian'):
                            mod_RightWidth = Ipl_Image.shape[1]
                        elif (bb_MidR == CropRight):
                            mod_RightWidth = int(CropRight)
                            df = df.replace(BB_max_X2, CropRight) 
                        elif (bb_MidR > CropRight):
                            mod_RightWidth = int(CropRight)
                            df = df.replace(BB_max_X2, 0.0)
                            df = df.replace([index_max[0]], 0.0)
                        else:
                            mod_RightWidth = 0.0 #Check with Medha
                    else:
                         df = df.replace(BB_max_X2, CropRight)
                         mod_RightWidth = int(CropRight)
             else:
                 mod_RightWidth = 0.0 #Check with Medha
                            
         else:
             mod_LeftWidth = int(CropLeft)
             mod_RightWidth = int(CropRight)  
           
         for counter in range(0, (df.shape[0])):
            if (Labeltxt[counter][0] != CropLeft) and (Labeltxt[counter][0] != 0):
                Labeltxt[counter][0] = Labeltxt[counter][0] - CropLeft
                if(Labeltxt[counter][0] < 1):
                    Labeltxt[counter][0] = 0.0     
        
         for counter in range(0, (df.shape[0])):
            if(Labeltxt[counter][2] != CropRight) and (Labeltxt[counter][2] != 0):
                Labeltxt[counter][2] = Labeltxt[counter][2] - CropRight
                if (Labeltxt[counter][2] < 1):
                    Labeltxt[counter][2] = 0.0
                                
     else:
        mod_LeftWidth = 0.0
        mod_RightWidth = Ipl_Image.shape[1]
        
     Labeltxt = df.values.tolist()
     return Labeltxt, mod_LeftWidth, mod_RightWidth
 
       
"""--------------------------------------------------------------------------------------------
Calculate cropping height Function
--------------------------------------------------------------------------------------------"""   
def CropImage_Height(Ipl_Image, Labeltxt):
     Height_ratio = round((Ipl_Image.shape[0] / Imae_height), 2)
     CropTop    = 0 + ((Ipl_Image.shape[0] / 2) * Crop_percent)
     CropBottom  = int(Ipl_Image.shape[0]) - ((Ipl_Image.shape[0] / 2) * Crop_percent)

     dfh = pd.DataFrame(Labeltxt, columns=['X1','Y1','X2','Y2','Class'])
     
     BB_min_Y1 = dfh['Y1'].min()
     BB_max_Y2 = dfh['Y2'].max()
     for counter in range(0,(dfh.shape[0])):
        if Labeltxt[counter][0] == 0:
            Labeltxt[counter][1]= 0.0
            Labeltxt[counter][2]= 0.0
            Labeltxt[counter][3]= 0.0
     if(Height_ratio > 0.33):
         if(float(BB_max_Y2) > CropBottom) or (float(BB_min_Y1) < CropTop):
             if(BB_min_Y1 != 0):
                 bb_MidT = ((BB_min_Y1 - BB_min_Y1)/2) + BB_min_Y1 
                 if(bb_MidT == CropTop):
                     mod_TopHeight = 0.0
                 else:
                     mod_TopHeight = int(CropTop)
             else:
                  mod_TopHeight = 0.0
                 
             if (BB_max_Y2 != 0):
                    bb_MidB = ((BB_max_Y2 - BB_max_Y2) / 2) + BB_max_Y2
                    if (bb_MidB == CropBottom):
                        mod_BottomHeight = Ipl_Image.shape[0] 
                    else:
                        mod_BottomHeight = int(CropBottom) 
             else:
                  mod_BottomHeight = Ipl_Image.shape[0]                             
             
         elif(float(BB_max_Y2) <= CropBottom) or (float(BB_min_Y1) >= CropTop):
            if (BB_max_Y2 != 0) or (BB_min_Y1 != 0):
                mod_TopHeight = int(CropTop)
                mod_BottomHeight = int(CropBottom)
                
               
         for counter in range(0, (dfh.shape[0])):
            if (Labeltxt[counter][1] != 0):
                Labeltxt[counter][1] = Labeltxt[counter][1] - CropTop
                if(Labeltxt[counter][1] < 1):
                    Labeltxt[counter][1] = 0.0     
        
         for counter in range(0, (dfh.shape[0])):
            if(Labeltxt[counter][3] != 0):
                Labeltxt[counter][3] = Labeltxt[counter][3] - CropBottom
                if (Labeltxt[counter][3] < 1):
                    Labeltxt[counter][3] = 0.0
                                
     else:
        mod_TopHeight = 0.0
        mod_BottomHeight = Ipl_Image.shape[0]
        
     return Labeltxt, mod_TopHeight, mod_BottomHeight            
         
   
def Aspect_Ratio(Ipl_Image, Ipl_Aspect, Labeltxt):
    if (Ipl_Aspect >= 1.6 and Ipl_Aspect <= 1.8):
        resizedImage = cv2.resize(Ipl_Image,(Image_width,Imae_height))
        bb_info = resizeImgBB(Ipl_Image, resizedImage,Labeltxt, shape )
        image_status = Check_BBprecent(resizedImage,bb_info)
        elseStatus = 0
        if Debug:
            print ("image_status in If Loop",image_status )

    elif(Ipl_Aspect>= 3.0 and Ipl_Aspect<= 3.4):
        resizedImage = cv2.resize(Ipl_Image,(Image_width,Imae_height1))
        bb_info = resizeImgBB(Ipl_Image, resizedImage, Labeltxt, shape)
        image_status = Check_BBprecent(resizedImage,bb_info)
        elseStatus = 0
        if Debug:
            print ("image_status in Elif Loop",image_status )
        
    else:
        bb_info = []
        for cand in range(0, shape[0]):
            bb = [Labeltxt[cand][4], Labeltxt[cand][5], Labeltxt[cand][6], Labeltxt[cand][7], Labeltxt[cand][0]]
            bb_info.append([bb])
            elseStatus = 1
        if Debug:
            print ("image_status in else Loop",image_status )

        image_status = Check_BBprecent(Ipl_Image,bb_info)      
        bb_info, mod_LeftWidth, mod_RightWidth   = CropImage_Width(Ipl_Image,bb_info)
        bb_info, mod_TopHeight, mod_BottomHeight = CropImage_Height(Ipl_Image, bb_info)
        resizedImage = Ipl_Image[mod_TopHeight : mod_BottomHeight, mod_LeftWidth : mod_RightWidth]
    
    return resizedImage, image_status, elseStatus, bb_info

    
"""--------------------------------------------------------------------------------------------
Main cropping width and height Function
--------------------------------------------------------------------------------------------"""     
for ImageName in ImagePath:
    name = ImageName.split("\\")[-1].split(".")[0]
    print ("\nImage_Name:  ",ImageName)
    for labelName in labeltxt:
        label = labelName.split("\\")[-1].split(".")[0]
        if (name == label):
            Labeltxt = np.genfromtxt(labelName, dtype=np.str)
            if Labeltxt.shape[0] != 0:
                Ipl_Image = cv2.imread(ImageName)
                Ipl_Aspect = Ipl_Image.shape[1] / Ipl_Image.shape[0]
                if (len(Labeltxt.shape) == 1):
                    shape = (1,(Labeltxt.shape[0]))
                    Labeltxt = [Labeltxt]
                else:
                    shape = ((Labeltxt.shape[0]),(Labeltxt.shape[1]))
                    
                TempLabeltxt = Labeltxt
                resizedImage, image_status, elseStatus, bb_info = Aspect_Ratio(Ipl_Image, Ipl_Aspect, Labeltxt)
                if elseStatus == 1:
                    resizedImage, image_status, elseStatus, bb_info = Aspect_Ratio(Ipl_Image, Ipl_Aspect, Labeltxt)
                    
                if image_status == 1:
                    #Convert Data to Kiiti format
                    Kitti = pd.DataFrame(TempLabeltxt, columns=['Class','Trunc', 'Occu', 'Alpha', 'X1','Y1','X2','Y2','Dimh','Dimw', 'Diml','CamY','CamY','CamZ','Rot'])
                    bb_format = pd.DataFrame(bb_info, columns=['X1','Y1','X2','Y2','Class'])
                    Kitti['X1'] = bb_format['X1']
                    Kitti['Y1'] = bb_format['Y1']
                    Kitti['X2'] = bb_format['X2']
                    Kitti['Y2'] = bb_format['Y2']
                    bb_info = Kitti.values.tolist()
                    
                    if Debug:
                        Resized_VisImage = resizedImage
                        for bb in range(0,len(bb_info)):    
                            p1        = ((int(float(bb_info[bb][4]))),int(float(bb_info[bb][5])))
                            p2        = ((int(float(bb_info[bb][6]))),int(float(bb_info[bb][7])))
                            color     = (0,0,255)
                            thickness = 1
                            textPos = (p1[0],p2[1]+6) 
                            OccPos = (p2[0],p1[1]-6)
                            cv2.rectangle(Resized_VisImage, p1, p2, color, thickness)
                            cv2.putText(Resized_VisImage,bb_info[bb][0],textPos, font, 0.3,(0,255,0),1,cv2.LINE_AA)
                            
                    #Data Augmentation Function
                    AugData = aug.data_augmentation(resizedImage,bb_info )
                    
                    if Dump_AugData:
                        #Dump data 
                        aug.DumpData(name, AugData)
                
                if Debug:
                    DisplayImage("VISUALIZER", Resized_VisImage) 
                if Dump_ResizedData:
                    SaveImage(name, Resized_VisImage)
                    
    if Debug:   
        k = cv2.waitKey(0) & 0xFF
        if (k == ord('q')) or (k == ord('Q')):
            break
            
cv2.destroyAllWindows()
                    
                                      
  

