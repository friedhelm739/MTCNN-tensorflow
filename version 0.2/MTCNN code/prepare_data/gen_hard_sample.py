# -*- coding: utf-8 -*-
"""
@author: friedhelm

"""
import sys
sys.path.append("../")

from detection.mtcnn_detector import MTCNN_Detector
from core.MTCNN_model import Pnet_model,Rnet_model,Onet_model
from core.tool import IoU
import cv2
import numpy as np
import os
import time
import argparse

def arg_parse():
    
    parser=argparse.ArgumentParser()
    
    parser.add_argument("--img_size",default=12,type=int, help='img size to generate')
    parser.add_argument("--base_dir",default="../",type=str, help='base path to save TFRecord file')
    parser.add_argument("--threshold",default=[0.8,0.8,0.6],type=list, help='threshold of models')
    parser.add_argument("--min_face_size",default=10,type=int, help='min face size')
    parser.add_argument("--factor",default=0.79,type=int, help='factor of img')
    parser.add_argument("--model_path",type=list, help='model path')
    parser.add_argument("--model_name",default="Pnet",type=str, help='from which model to generate data')   
    
    return parser


def main():
    
    if(model_name in ["Pnet","Rnet","Onet"]):
        model[0]=Pnet_model
    if(model_name in ["Rnet","Onet"]):
        model[1]=Rnet_model
    if(model_name=="Onet"):
        model[2]=Onet_model
        
    detector=MTCNN_Detector(model,model_path,batch_size,factor,min_face_size,threshold)

    with open(WIDER_spilt_dir) as filenames:
        p=0
        neg_idx=0
        pos_idx=0
        par_idx=0
        idx=0
        for line in filenames.readlines():
            line=line.strip().split(' ')
            if(p==0):
                pic_dir=line[0]
                p=1
                boxes=[]
            elif(p==1):
                k=int(line[0])
                p=2
            elif(p==2):
                b=[]            
                k=k-1
                if(k==0):
                    p=0                
                for i in range(4):
                    b.append(int(line[i]))
                boxes.append(b)
                # format of boxes is [x,y,w,h]
                if(p==0):

                    img=cv2.imread(os.path.join(WIDER_dir,pic_dir))

                    face_box,_=detector.detect_single_face(img)
                    neg_num=0
                    for t_box in face_box:

                        ti_box=t_box.copy()
                        ti_box=[int(_) for _ in ti_box]   
                        
                        Iou=IoU(np.array(t_box),np.array(boxes))

                        if((np.max(Iou)<0.3)&(neg_num<60)):
                            resized_img = cv2.resize(img[ti_box[1]:ti_box[3],ti_box[0]:ti_box[2],:], (img_size,img_size))
                            cv2.imwrite(os.path.join(negative_dir,'neg_%d.jpg'%(neg_idx)),resized_img)
                            f2.write(os.path.join(negative_dir,'neg_%d.jpg'%(neg_idx)) + ' 0\n')
                            neg_idx=neg_idx+1
                            neg_num=neg_num+1

                        else:
                            x1,y1,w1,h1=boxes[np.argmax(Iou)]

                            offset_x1 = (x1 - t_box[0]) / float(t_box[2]-t_box[0]+1)
                            offset_y1 = (y1 - t_box[1]) / float(t_box[3]-t_box[1]+1)
                            offset_x2 = (x1+w1 - t_box[2]) / float(t_box[2]-t_box[0]+1)
                            offset_y2 = (y1+h1 - t_box[3]) / float(t_box[3]-t_box[1]+1)                         

                            if(np.max(Iou)>0.65):                    
                                resized_img = cv2.resize(img[ti_box[1]:ti_box[3],ti_box[0]:ti_box[2],:], (img_size, img_size))
                                cv2.imwrite(os.path.join(positive_dir,'pos_%d.jpg'%(pos_idx)),resized_img)
                                f1.write(os.path.join(positive_dir,'pos_%d.jpg'%(pos_idx)) + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1,offset_y1,offset_x2,offset_y2))
                                pos_idx=pos_idx+1   

                            elif(np.max(Iou)>0.4):
                                resized_img = cv2.resize(img[ti_box[1]:ti_box[3],ti_box[0]:ti_box[2],:], (img_size, img_size))
                                cv2.imwrite(os.path.join(par_dir,'par_%d.jpg'%(par_idx)),resized_img)
                                f3.write(os.path.join(par_dir,'par_%d.jpg'%(par_idx)) + ' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1,offset_y1,offset_x2,offset_y2))                           
                                par_idx=par_idx+1
                    idx+=1
                    if(idx%100==0):
                        print('idx: ',idx," ;neg_idx: ",neg_idx," ;pos_idx: ",pos_idx," ;par_idx: ",par_idx)
                        print(time.time()-begin)
    print("pics all done,neg_pics %d in total,pos_pics %d in total,par_pics %d in total"%(neg_idx,pos_idx,par_idx))

if __name__=="__main__":
    
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   
    
    parser=arg_parse()
    base_dir=parser.base_dir
    img_size=parser.img_size    
    threshold=parser.threshold 
    min_face_size=parser.min_face_size 
    factor=parser.factor    
    model_name=parser.model_name 
    model_path=parser.model_path     
      
    #parameters to modify
    #threshold=[0.8,0.8,0.6]
    #min_face_size=10
    #base=200000    
    #img_size=12    
    #base_dir="/home/dell/Desktop/MTCNN"
    #factor=0.79      
    #model_name="Pnet"
    #model_path=[os.path.join(base_dir,"model/Pnet_model/Pnet_model.ckpt-20000"),
    #            None,
    #            None] 
    #parameters to modify
    batch_size=1
    model=[None,None,None]
    
    WIDER_dir=os.path.join(base_dir,"prepared_data/WIDER_train/images")
    WIDER_spilt_dir=os.path.join(base_dir,"prepared_data/wider_face_split/wider_face_train_bbx_gt.txt")
    negative_dir=os.path.join(base_dir,"DATA/%d/negative"%(img_size))
    positive_dir=os.path.join(base_dir,"DATA/%d/positive"%(img_size))
    par_dir=os.path.join(base_dir,"DATA/%d/part"%(img_size))
    save_dir=os.path.join(base_dir,"DATA/%d"%(img_size))
    
    if not os.path.exists(positive_dir):
        os.makedirs(positive_dir)
    if not os.path.exists(par_dir):
        os.makedirs(par_dir)
    if not os.path.exists(negative_dir):
        os.makedirs(negative_dir)    
    
    begin=time.time()
    
    f1 = open(os.path.join(save_dir, 'pos_%d.txt'%(img_size)), 'w')
    f2 = open(os.path.join(save_dir, 'neg_%d.txt'%(img_size)), 'w')
    f3 = open(os.path.join(save_dir, 'par_%d.txt'%(img_size)), 'w') 
    
    main()
    
    f1.close()
    f2.close()
    f3.close()      

    print(time.time()-begin)    