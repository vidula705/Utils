# -*- coding: utf-8 -*-
"""
Created on Tue May 30 16:48:57 2017

@author: vidula
"""
#File 2 : To read roi image and calculate grad, vertical and horizontal sum and dump the array of roi
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

vdumpDir = './graph/'
v_perc_ref = 0.20
NoiseTh_H = 140 #in edge pix
NoiseTh_L = 80 #in edge pix
imgSizeTh = 130 #pix. >150 is near and <150 is far
ScaleLimit = 512
PLOT_VERTICAL_GRAPH = 0
DEBUG_VERTICAL_PROFILING = 0
frameWidth = 1280
vertLimit2_perc = 0.30
debug = 1
#######################################################################################################
def Edge_operator(roi, imgHt):
    v_kernel = np.array([1,0,-1, 1,0,-1, 2,0,-2, 1,0,-1, 1,0,-1]).reshape(5,3) #vertical kernel  
    y_grad = cv2.filter2D(roi, cv2.CV_32F, v_kernel, None, (-1,-1), 0, cv2.BORDER_DEFAULT)
    y_grad = np.abs(y_grad.copy()).astype(np.int) #vertical  
    y_grad[0:2,0:y_grad.shape[1]] = 0
    y_grad[y_grad.shape[0]-2:,0:y_grad.shape[1]] = 0

    if imgHt>=imgSizeTh:
        y_grad[y_grad<NoiseTh_H] = 0 # Just an expt
    else:
        y_grad[y_grad<NoiseTh_L] = 0 # Just an expt

    return y_grad

 ####################################################################################################### 
def vertical_profile(edgeImg1, processMode, seedPtX, seedPtY):
    y_grad_arrayV = np.array(edgeImg1)
    y_grad_arrayV = y_grad_arrayV.sum(axis = 0) # 0= vertical sum 1= horizontal sum

    if DEBUG_VERTICAL_PROFILING == 1:
        if not os.path.exists('./graph/vsum_py.csv'):
            np.savetxt('./vsum_py.csv',np.reshape(y_grad_arrayV,[1,y_grad_arrayV.shape[0]]), fmt='%d',delimiter=',')
            
    max_Location = np.argmax(y_grad_arrayV)
    if debug == 1:
        print "Max_Location of X_Grad:", max_Location
    if processMode == 0:
        v_perc = v_perc_ref
    elif processMode == 1:
        v_perc = 0.50
    else:
        v_perc = 0.20
    
    ExtremePt = edgeImg1.shape[1]/3   
    x1 = max_Location - ExtremePt
    x2 = max_Location + ExtremePt
    tempLimitx1 = max_Location - ExtremePt
    tempLimitx2 = max_Location + ExtremePt
    
    Limitx1 = max_Location - ExtremePt

    if Limitx1 < 0:
        Limitx1 = 0
    for i in range(max_Location, Limitx1,-1):
        if(y_grad_arrayV[i] <= y_grad_arrayV[max_Location]*v_perc):
            x1 = i
            break
        else :x1 = Limitx1
    Limitx2 = max_Location + ExtremePt
    if Limitx2 >= y_grad_arrayV.shape[0]:
        Limitx2 = y_grad_arrayV.shape[0] - 1
    
    for j in range(max_Location, Limitx2 ): 
        if(y_grad_arrayV[j] <= y_grad_arrayV[max_Location]*v_perc):
            x2 = j
            break
        else :x2 = Limitx2
#####################################################################    
    if 1:
        if debug == 1:
           print 'ProcessMode',processMode,'Before Region Extension', "XLEFT:", x1,"XRIGHT:", x2
        if seedPtY > ScaleLimit:
            if (seedPtX < frameWidth/2):
                if tempLimitx1==x1 or tempLimitx1 < 0:
                    tempLimitx1 = 0
                if x1 != 0:
                   NewMaxLocationLeft = np.argmax(y_grad_arrayV[tempLimitx1:x1])+tempLimitx1
                   
                else: 
                    NewMaxLocationLeft = max_Location
                NewLimitx1 = NewMaxLocationLeft - ExtremePt
                if NewLimitx1 < 0:
                    NewLimitx1 = 0
                for i in range(NewMaxLocationLeft, NewLimitx1,-1):
                    if(y_grad_arrayV[i] <= y_grad_arrayV[NewMaxLocationLeft]*vertLimit2_perc):
                        x1 = i
                        break
            else:
                NewMaxLocationRight = np.argmax(y_grad_arrayV[x2:tempLimitx2])+ x2
               
                NewLimitx2 = NewMaxLocationRight + ExtremePt
                if NewLimitx2 >= y_grad_arrayV.shape[0]:
                    NewLimitx2 = y_grad_arrayV.shape[0] - 1
                for j in range(NewMaxLocationRight,NewLimitx2 ):  
                    if(y_grad_arrayV[j] <= y_grad_arrayV[NewMaxLocationRight]*vertLimit2_perc):
                        x2 = j
                        break    
        if debug == 1:
           print 'ProcessMode',processMode,'After Region Extension',"XLEFT:", x1,"XRIGHT:", x2
        
##########################################################################        
    v_roi = edgeImg1[0 : edgeImg1.shape[0], x1 : x2] 

    plt.close('all')
    if PLOT_VERTICAL_GRAPH == 1:
        plt.plot(y_grad_arrayV, 'k')
        plt.plot([max_Location, max_Location], [plt.ylim()[0], plt.ylim()[1]], 'k-')
        plt.plot([x1, x1], [plt.ylim()[0], plt.ylim()[1]], 'k-')
        plt.plot([x2, x2], [plt.ylim()[0], plt.ylim()[1]], 'k-')
        #plt.savefig(vdumpDir+imageName1+'_'+str(cnt)+'_edge_vsum1_%d.png' %(processMode))
      
    return v_roi

#######################################################################################################
def horizontal_profile(roi):
    y_grad_array = np.array(roi)
    y_grad_array = y_grad_array.sum(axis = 1) # 0= vertical sum 1= horizontal sum
    return y_grad_array


   
