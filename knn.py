# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 14:18:25 2018

@author: vidula
"""

from os import listdir
from os.path import isfile, join
#import argparse
import numpy as np
import sys
import os
#import shutil
import random 
import glob

labeltxt  = glob.glob("F:\\DL\\BigData\\DataAug\\lab\\*.txt")
clusters = 3
output_dir = './/anchors'

width_in_cfg_file = 416.
height_in_cfg_file = 416.

"""--------------------------------------------------------------------------------------------
IOU Function
--------------------------------------------------------------------------------------------"""     

def IOU(x,centroids):
    similarities = []
    for centroid in centroids:
        c_w,c_h = centroid
        w,h = x
        if c_w>=w and c_h>=h:
            similarity = w*h/(c_w*c_h)
        elif c_w>=w and c_h<=h:
            similarity = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w<=w and c_h>=h:
            similarity = c_w*h/(w*h + c_w*(c_h-h))
        else: #means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity) # will become (k,) shape
    return np.array(similarities) 

"""--------------------------------------------------------------------------------------------
Average IOU Function
--------------------------------------------------------------------------------------------"""     
def avg_IOU(X,centroids):
    n,d = X.shape
    sum = 0.
    for i in range(X.shape[0]):
        #note IOU() will return array which contains IoU for each centroid and X[i] 
        sum+= max(IOU(X[i],centroids)) 
    return sum/n

"""--------------------------------------------------------------------------------------------
Write anchors to file
--------------------------------------------------------------------------------------------"""     
def write_anchors_to_file(centroids,X,anchor_file):
    f = open(anchor_file,'w')   
    anchors = centroids.copy()
    for i in range(anchors.shape[0]):
        anchors[i][0]*=width_in_cfg_file/32.
        anchors[i][1]*=height_in_cfg_file/32.
         
    widths = anchors[:,0]
    sorted_indices = np.argsort(widths)

    print ('Anchors = ', anchors[sorted_indices] )
        
    for i in sorted_indices[:-1]:
        f.write('%0.2f,%0.2f, '%(anchors[i,0],anchors[i,1]))

    #there should not be comma after last anchor, that's why
    f.write('%0.2f,%0.2f\n'%(anchors[sorted_indices[-1:],0],anchors[sorted_indices[-1:],1]))
    
    f.write('%f\n'%(avg_IOU(X,centroids)))

"""--------------------------------------------------------------------------------------------
Kmean Clutering Function
--------------------------------------------------------------------------------------------"""     
def kmeans(X,centroids,eps,anchor_file):
    if (len(centroids.shape) == 1):
        centroids.shape = ((centroids.shape[0],1))
    N = X.shape[0]
    k,dim = centroids.shape
    prev_assignments = np.ones(N)*(-1)    
    iter = 0
    old_D = np.zeros((N,k))

    while True:
        D = [] 
        iter+=1           
        for i in range(N):
            d = 1 - IOU(X[i],centroids)
            D.append(d)
        D = np.array(D) # D.shape = (N,k)
        
        print ("iter {}: dists = {}".format(iter,np.sum(np.abs(old_D-D))))
            
        #assign samples to centroids 
        assignments = np.argmin(D,axis=1)
        
        if (assignments == prev_assignments).all() :
            print ("Centroids = ",centroids            )
            write_anchors_to_file(centroids,X,anchor_file)
            return

        #calculate new centroids
        centroid_sums=np.zeros((k,dim),np.float)
        for i in range(N):
            centroid_sums[assignments[i]]+=X[i]        
        for j in range(k):            
            centroids[j] = centroid_sums[j]/(np.sum(assignments==j))
        
        prev_assignments = assignments.copy()     
        old_D = D.copy()  

"""--------------------------------------------------------------------------------------------
IOU Function
--------------------------------------------------------------------------------------------"""     

def main():
    #parser = argparse.ArgumentParser()
    #parser.add_argument('-filelist', default = 'F:\\DL\\BigData\\DataAug\\lab\\000000.txt', 
     #                   help='path to filelist\n' )
    #parser.add_argument('-output_dir', default = './/anchors', type = str, 
     #                   help='Output anchor directory\n' )  
    #parser.add_argument('-num_clusters', default = 3, type = int, 
     #                   help='number of clusters\n' )  

    num_clusters = clusters
    annotation_dims = []
    for file_path in labeltxt:
        label = (np.genfromtxt(file_path, dtype = np.str))
        for i in range (0, len(label)):
            #print (data1[i][4])
            w =(float(label[i][6]) - float(label[i][4])) 
            h =(float(label[i][7]) - float(label[i][5]))
            annotation_dims.append((w,h))
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    #print (annotation_dims)
  #  for label in labelList:
       # i = i+1
        #print(i)
      #  w =(float(label[6]) - float(label[4])) 
       # h =(float(label[7]) - float(label[5])) 
        #annotation_dims.append((w,h))
        #print (w,h)
        #sys.exit()
    #g = 98.33
    #v = 164.92
    #for line in range(0,20):
        #line = line.rstrip('\n')
     #   w,h = g,v          
        #print (w,h)
       # annotation_dims.append((w,h))
      #  g, v = (g+3,v+3) 
    #annotation_dims.append(map(float,(98.33,164.92)))
    annotation_dims = np.array(annotation_dims)
    #print ("annotation_dims",annotation_dims.shape)
    eps = 0.005
    
    if num_clusters == 0:
        for num_clusters in range(1,11): #we make 1 through 10 clusters 
            anchor_file = join( output_dir,'anchors%d.txt'%(num_clusters))

            indices = [ random.randrange(annotation_dims.shape[0]) for i in range(num_clusters)]
            centroids = annotation_dims[indices]
            kmeans(annotation_dims,centroids,eps,anchor_file)
            print ('centroids.shape', centroids.shape)
    else:
        anchor_file = join( output_dir,'anchors%d.txt'%(num_clusters))
        #print (anchor_file)
        
        indices = [ random.randrange(annotation_dims.shape[0]) for i in range(num_clusters)]
        centroids = annotation_dims[indices]
        print ("centroids",centroids)
        
        kmeans(annotation_dims,centroids,eps,anchor_file)
      
if __name__=="__main__":
    main()
    #main(sys.argv)
