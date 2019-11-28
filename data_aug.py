"""
Created on Thu Aug  3 12:05:39 2018

@author: vidula
"""

from imgaug import augmenters as iaa
import imgaug as ia
import os
import cv2

#----------------------------------Verticle Flip-------------------------------------------------------------------"""
seq_vflip = iaa.Sequential([iaa.Flipud(1)])


#----------------------------------Horizantal Flip ---------------------------------------------------------------------"""
seq_hflip = iaa.Sequential([iaa.Fliplr(1)])


#"""----------------------------------Affine Rotate-------------------------------------------------------------------"""
seq_affine_rotate = iaa.Sequential([iaa.Affine(rotate=15)])


#"""----------------------------------Affine Scale -------------------------------------------------------------------"""
seq_affine_scale = iaa.Sequential([iaa.Affine(scale=(0.5, 0.7))])


#"""----------------------------------GaussianBlur -------------------------------------------------------------------"""
seq_GaussianBlur = iaa.Sequential([iaa.GaussianBlur(sigma=(1.0, 3.0))])  # blur images with a sigma of 0 to 3.0


#"""----------------------------------ContrastNormilazation-----------------------------------------------------------"""
seq_ContrastNormalization = iaa.Sequential([iaa.ContrastNormalization((0.75, 1.5))])


#"""----------------------------------AdditiveGaussianNoise-----------------------------------------------------------"""
seq_AdditiveGaussianNoise = iaa.Sequential([iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)])


#"""----------------------------------Sharpen-------------------------------------------------------------------------"""
seq_Sharpen = iaa.Sequential([iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))])


#"""----------------------------------Multiply------------------------------------------------------------------------"""
seq_Multiply = iaa.Sequential([iaa.Multiply((0.5, 1.5), per_channel=0.5)])


#"""----------------------------------Affine Translate----------------------------------------------------------------"""
seq_Affine_translate_percent = iaa.Sequential([iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})])


"""---------------------------------------------------------------------------------------------------------------------
Function to store the augmented frames in folder
---------------------------------------------------------------------------------------------------------------------"""
def DumpData(ImageName, DataAugImages):
    Dump_path = "./AugmentData"
    if not os.path.exists(Dump_path):
        os.makedirs(Dump_path)

    for counter in DataAugImages.keys():
        Imagedata = DataAugImages[counter][0]
        BBData    = DataAugImages[counter][1]
        aug_img_path = Dump_path + "/" + counter + "/" + "Images/"
        if not os.path.exists(aug_img_path):
            os.makedirs(aug_img_path)
        img_writepath = os.path.join(aug_img_path, ImageName +".png")
        
        cv2.imwrite(img_writepath,Imagedata)
        aug_label_path = Dump_path + "/" + counter + "/" + "Labels/"
        if not os.path.exists(aug_label_path):
            os.makedirs(aug_label_path)
        lab_writepath = os.path.join(aug_label_path, ImageName +".txt")

        labfile =  open(lab_writepath, 'w') 
        for bb in BBData:    
            bb =  ((str(bb)[1:-1]).replace(",", "")).replace("'","")
            labfile.write(str(bb) + '\n') 
        labfile.close()


"""---------------------------------------------------------------------------------------------------------------------
                                Main data augmentation Logic
---------------------------------------------------------------------------------------------------------------------"""
ia.seed(1)
def data_augmentation(Ipl_Image, Label):
        #Lists to store data after data augmentation
        vflip                 = []
        hflip                 = []
        affine_rotate         = []
        affine_scale          = []
        GaussianBlur          = []
        ContrastNormalization = []
        AdditiveGaussianNoise = []
        Sharpen               = []
        Multiply              = []
        Affine_translate_percent = []
        
        seq_det = seq_vflip.to_deterministic()
        image_aug_vflip = seq_det.augment_images([Ipl_Image])[0]
        
        seq_det = seq_hflip.to_deterministic()
        image_aug_hflip = seq_det.augment_images([Ipl_Image])[0]
        
        seq_det = seq_affine_rotate.to_deterministic()
        image_aug_affine_rotate = seq_det.augment_images([Ipl_Image])[0]       
        
        seq_det = seq_affine_scale.to_deterministic()
        image_aug_affine_scale = seq_det.augment_images([Ipl_Image])[0]
        
        seq_det = seq_GaussianBlur.to_deterministic()
        image_aug_GaussianBlur = seq_det.augment_images([Ipl_Image])[0]        
        
        seq_det = seq_ContrastNormalization.to_deterministic()
        image_aug_ContrastNormalization = seq_det.augment_images([Ipl_Image])[0]
        
        seq_det = seq_AdditiveGaussianNoise.to_deterministic()
        image_aug_AdditiveGaussianNoise = seq_det.augment_images([Ipl_Image])[0]
        
        seq_det = seq_Sharpen.to_deterministic()
        image_aug_Sharpen = seq_det.augment_images([Ipl_Image])[0] 
        
        seq_det = seq_Multiply.to_deterministic()
        image_aug_Multiply = seq_det.augment_images([Ipl_Image])[0]   
                
        seq_det = seq_Affine_translate_percent.to_deterministic()
        image_aug_Affine_translate_percent = seq_det.augment_images([Ipl_Image])[0]
        
                
        for cand in range(0, len(Label)):
            if (Label[cand][4] != 0 or Label[cand][5] != 0 or Label[cand][6] != 0 or Label[cand][7] != 0):
                label = ((Label[cand][0]))

                trunc = (float(Label[cand][1]))
                Occul = (float(Label[cand][2]))
                alpha = (float(Label[cand][3]))
                x1 = (int(Label[cand][4]))
                y1 = (int(Label[cand][5]))
                x2 = (int(Label[cand][6]))
                y2 = (int(Label[cand][7]))
                Dimh = (float(Label[cand][8]))
                Dimw = (float(Label[cand][9]))
                Diml = (float(Label[cand][10]))
                CamX= (float(Label[cand][11]))
                CamY = (float(Label[cand][12]))
                CamZ = (float(Label[cand][13]))
                Rot = (float(Label[cand][14]))
                
                BB = [ia.BoundingBox(x1, y1, x2, y2, label)]
                bbs = ia.BoundingBoxesOnImage(BB, shape=Ipl_Image.shape)             

                bbs_aug_vfilp = seq_det.augment_bounding_boxes([bbs])[0]
                after_vflip = bbs_aug_vfilp.bounding_boxes
                coord = [label,trunc, Occul,alpha,after_vflip[0].x1, after_vflip[0].y1, after_vflip[0].x2, after_vflip[0].y2,
                         Dimh, Dimw,Diml,CamX, CamY, CamZ, Rot   ]
                vflip.append(coord)
                
                bbs_aug_hflip = seq_det.augment_bounding_boxes([bbs])[0]
                after_hflip=(bbs_aug_hflip.bounding_boxes)
                temp = [label,trunc, Occul,alpha,after_hflip[0].x1, after_hflip[0].y1, after_hflip[0].x2, after_hflip[0].y2,
                        Dimh, Dimw,Diml,CamX, CamY, CamZ,Rot ]
                hflip.append(temp)
                
                bbs_aug_affine_rotate = seq_det.augment_bounding_boxes([bbs])[0]
                after_affine_rotate=(bbs_aug_affine_rotate.bounding_boxes)
                temp1 = [label,trunc, Occul,alpha,after_affine_rotate[0].x1, after_affine_rotate[0].y1, after_affine_rotate[0].x2,after_affine_rotate[0].y2,
                         Dimh, Dimw,Diml,CamX, CamY, CamZ,Rot ]
                affine_rotate.append(temp1) 

                bbs_aug_affine_scale = seq_det.augment_bounding_boxes([bbs])[0]
                after_affine_scale=(bbs_aug_affine_scale.bounding_boxes)
                temp2 = [label,trunc, Occul,alpha,after_affine_scale[0].x1, after_affine_scale[0].y1,after_affine_scale[0].x2, after_affine_scale[0].y2,
                         Dimh, Dimw,Diml,CamX, CamY, CamZ,Rot ]
                affine_scale.append(temp2)
    
                bbs_aug_GaussianBlur = seq_det.augment_bounding_boxes([bbs])[0]
                after_GaussianBlur=(bbs_aug_GaussianBlur.bounding_boxes)
                temp3 =[label,trunc, Occul,alpha,after_GaussianBlur[0].x1, after_GaussianBlur[0].y1, after_GaussianBlur[0].x2,after_GaussianBlur[0].y2,
                        Dimh, Dimw,Diml,CamX, CamY, CamZ,Rot ]
                GaussianBlur.append(temp3)
    
                bbs_aug_ContrastNormalization = seq_det.augment_bounding_boxes([bbs])[0]
                after_ContrastNormalization=(bbs_aug_ContrastNormalization.bounding_boxes)
                temp4 = [label,trunc, Occul,alpha,after_ContrastNormalization[0].x1, after_ContrastNormalization[0].y1,after_ContrastNormalization[0].x2, after_ContrastNormalization[0].y2, 
                         Dimh, Dimw,Diml,CamX, CamY, CamZ,Rot ]
                ContrastNormalization.append(temp4)
    
                bbs_aug_AdditiveGaussianNoise = seq_det.augment_bounding_boxes([bbs])[0]
                after_AdditiveGaussianNoise=(bbs_aug_AdditiveGaussianNoise.bounding_boxes)
                temp5 = [label,trunc, Occul,alpha,after_AdditiveGaussianNoise[0].x1, after_AdditiveGaussianNoise[0].y1,after_AdditiveGaussianNoise[0].x2, after_AdditiveGaussianNoise[0].y2,
                         Dimh, Dimw,Diml,CamX, CamY, CamZ,Rot ]
                AdditiveGaussianNoise.append(temp5)
    
                bbs_aug_Sharpen = seq_det.augment_bounding_boxes([bbs])[0]
                after_Sharpen=(bbs_aug_Sharpen.bounding_boxes)
                temp6 = [label,trunc, Occul,alpha,after_Sharpen[0].x1, after_Sharpen[0].y1, after_Sharpen[0].x2, after_Sharpen[0].y2,
                         Dimh, Dimw,Diml,CamX, CamY, CamZ,Rot ]
                Sharpen.append(temp6)
    
                bbs_aug_Multiply = seq_det.augment_bounding_boxes([bbs])[0]
                after_Multiply=(bbs_aug_Multiply.bounding_boxes)
                temp7 =[label,trunc, Occul,alpha,after_Multiply[0].x1, after_Multiply[0].y1, after_Multiply[0].x2, after_Multiply[0].y2,
                        Dimh, Dimw,Diml,CamX, CamY, CamZ,Rot ]
                Multiply.append(temp7)
    
                bbs_aug_Affine_translate_percent = seq_det.augment_bounding_boxes([bbs])[0]
                after_Affine_translate_percent=(bbs_aug_Affine_translate_percent.bounding_boxes)
                temp8 =[label,trunc, Occul,alpha,after_Affine_translate_percent[0].x1,after_Affine_translate_percent[0].y1,after_Affine_translate_percent[0].x2,after_Affine_translate_percent[0].y2,
                        Dimh, Dimw,Diml,CamX, CamY, CamZ,Rot ]
                Affine_translate_percent.append(temp8)
                
        #Dictionary: Key as folder name, Image data and bb values
        AugData = {"ScaledImages" : [image_aug_affine_scale,affine_scale], "RotateImages" : [image_aug_affine_rotate,affine_rotate], "HFlipImages" : [image_aug_hflip,hflip],
             "VFlip_images" : [image_aug_vflip, vflip], "GBlurImages" : [image_aug_GaussianBlur, GaussianBlur], "ConNormImages" : [image_aug_ContrastNormalization, ContrastNormalization],
             "AdditivGNoiseImages" : [image_aug_AdditiveGaussianNoise, AdditiveGaussianNoise], "SharpenImages" : [image_aug_Sharpen, Sharpen], "MultiplyImages" : [image_aug_Multiply, Multiply],
             "AffineTranPerImages" : [image_aug_Affine_translate_percent, Affine_translate_percent]}

    
        return AugData
    
    
