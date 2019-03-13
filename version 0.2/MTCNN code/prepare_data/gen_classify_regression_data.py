# -*- coding: utf-8 -*-
"""
@author: friedhelm

"""
from core.tool import IoU
import numpy as np
from numpy.random import randint
import cv2
import os
import time

def main():

    f1 = open(os.path.join(save_dir, 'pos_%d.txt'%(img_size)), 'w')
    f2 = open(os.path.join(save_dir, 'neg_%d.txt'%(img_size)), 'w')
    f3 = open(os.path.join(save_dir, 'par_%d.txt'%(img_size)), 'w')    
    
    with open(WIDER_spilt_dir) as filenames:
        p=0
        neg_idx=0
        pos_idx=0
        par_idx=0
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
                    img=cv2.imread(os.path.join(WIDER_dir,pic_dir).replace('/','\\'))
                    h,w,c=img.shape
                    
                    #save num negative pics whose IoU less than 0.3
                    num=50
                    while(num):
                        size=randint(12,min(w,h)/2)
                        x=randint(0,w-size)
                        y=randint(0,h-size)
                        if(np.max(IoU(np.array([x,y,x+size,y+size]),np.array(boxes)))<0.3):
                            resized_img = cv2.resize(img[y:y+size,x:x+size,:], (img_size, img_size))
                            cv2.imwrite(os.path.join(negative_dir,'neg_%d.jpg'%(neg_idx)),resized_img)
                            f2.write(os.path.join(negative_dir,'neg_%d.jpg'%(neg_idx)) + ' 0\n')
                            neg_idx=neg_idx+1
                            num=num-1       

                    for box in boxes:
                        if((box[0]<0)|(box[1]<0)|(max(box[2],box[3])<20)|(min(box[2],box[3])<=5)): 
                            continue  
                        x1, y1, w1, h1 = box
                        
                        # crop images near the bounding box if IoU less than 0.3, save as negative samples
                        for i in range(10):
                            size = randint(12, min(w, h) / 2)
                            delta_x = randint(max(-size, -x1), w1)
                            delta_y = randint(max(-size, -y1), h1)
                            nx1 = int(max(0, x1 + delta_x))
                            ny1 = int(max(0, y1 + delta_y))
                            if((nx1 + size > w1)|(ny1 + size > h1)):
                                continue
                            if(np.max(IoU(np.array([nx1,ny1,nx1+size,ny1+size]),np.array(boxes)))<0.3):
                                resized_img = cv2.resize(img[y:y+size,x:x+size,:], (img_size, img_size))
                                cv2.imwrite(os.path.join(negative_dir,'neg_%d.jpg'%(neg_idx)),resized_img)
                                f2.write(os.path.join(negative_dir,'neg_%d.jpg'%(neg_idx)) + ' 0\n')
                                neg_idx=neg_idx+1
                                
                        #save num positive&part face whose IoU more than 0.65|0.4         
                        box_ = np.array(box).reshape(1, -1)
                        for i in range(10):
                            size=randint(np.floor(0.8*min(w1,h1)),np.ceil(1.25*max(w1,h1))+1)
                        
                            delta_w = randint(-w1 * 0.2, w1 * 0.2 + 1)
                            delta_h = randint(-h1 * 0.2, h1 * 0.2 + 1)
                            # random face box
                            nx1 = int(max(x1 + w1 / 2 + delta_w - size / 2, 0))
                            ny1 = int(max(y1 + h1 / 2 + delta_h - size / 2, 0))
                            nx2 = nx1 + size
                            ny2 = ny1 + size
                            
                            if( nx2 > w | ny2 > h):
                                continue 
                                
                            offset_x1 = (x1 - nx1) / float(size)
                            offset_y1 = (y1 - ny1) / float(size)
                            offset_x2 = (x1+w1 - nx2) / float(size)
                            offset_y2 = (y1+h1 - ny2) / float(size)                                
    
                            if(IoU(np.array([nx1,ny1,nx2,ny2]),box_)>0.65):                     
                                resized_img = cv2.resize(img[ny1:ny2,nx1:nx2,:], (img_size, img_size))
                                cv2.imwrite(os.path.join(positive_dir,'pos_%d.jpg'%(pos_idx)),resized_img)
                                f1.write(os.path.join(positive_dir,'pos_%d.jpg'%(pos_idx)) + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1,offset_y1,offset_x2,offset_y2))
                                pos_idx=pos_idx+1   
                                
                            elif(IoU(np.array([nx1,ny1,nx2,ny2]),box_)>0.4):
                                resized_img = cv2.resize(img[ny1:ny2,nx1:nx2,:], (img_size, img_size))
                                cv2.imwrite(os.path.join(par_dir,'par_%d.jpg'%(par_idx)),resized_img)
                                f3.write(os.path.join(par_dir,'par_%d.jpg'%(par_idx)) + ' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1,offset_y1,offset_x2,offset_y2))                           
                                par_idx=par_idx+1 
        print("pics all done,neg_pics %d in total,pos_pics %d in total,par_pics %d in total"%(neg_idx,pos_idx,par_idx))
        
    f1.close()
    f2.close()
    f3.close()  


if __name__=="__main__":
    
    img_size=12
    
    base_dir="E:\\friedhelm\\object\\face_detection_MTCNN"
    
    WIDER_dir=os.path.join(base_dir,"prepare_data\\WIDER_train\\images")
    WIDER_spilt_dir=os.path.join(base_dir,"prepare_data\\wider_face_split\\wider_face_train_bbx_gt.txt")
    negative_dir=os.path.join(base_dir,"DATA\\%d\\negative"%(img_size))
    positive_dir=os.path.join(base_dir,"DATA\\%d\\positive"%(img_size))
    par_dir=os.path.join(base_dir,"DATA\\%d\\part"%(img_size))
    save_dir=os.path.join(base_dir,"DATA\\%d"%(img_size))
    
    if not os.path.exists(positive_dir):
        os.makedirs(positive_dir)
    if not os.path.exists(par_dir):
        os.makedirs(par_dir)
    if not os.path.exists(negative_dir):
        os.makedirs(negative_dir) 
        
    begin=time.time()
    
    main()

    print(time.time()-begin)
#6841.530851840973