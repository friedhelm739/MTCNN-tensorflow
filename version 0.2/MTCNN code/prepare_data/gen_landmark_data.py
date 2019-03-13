# -*- coding: utf-8 -*-
"""
@author: friedhelm

"""

from core.tool import IoU,flip
import numpy as np
import random
from numpy.random import randint
import cv2
import os
import time

def main():
    
    f4 = open(os.path.join(save_dir, 'land_%d.txt'%(img_size)), 'w')

    with open(pic_spilt_dir) as filenames:
        land_idx=0
        for line in filenames:
            img_list=[]
            mark_list=[]
            line=line.strip().split(' ')
            img=cv2.imread(os.path.join(lfw_dir,line[0]))
            box=(line[1],line[2],line[3],line[4])
            box=[int(_) for _ in box]
            #format of box is [x,x+w,y,y+h]
            height,weight,channel=img.shape
            landmark=np.zeros((5,2))
            for i in range(5):
                mark=(float(line[5+2*i]),float(line[5+2*i+1]))
                landmark[i]=mark

            facemark=np.zeros((5,2))
            for i in range(5):
                mark=((landmark[i][0]-box[0])/(box[1]-box[0]),(landmark[i][1]-box[2])/(box[3]-box[2]))
                facemark[i]=mark
            img_list.append(cv2.resize(img[box[2]:box[3],box[0]:box[1]], (img_size, img_size)))  
            mark_list.append(facemark.reshape(10))

            box_=[box[0],box[2],box[1],box[3]]
            #format of box is [x,y,x+w,y+h]      
            x1,y1,x2,y2=box_
            w=x2-x1+1
            h=y2-y1+1

            if((x1<0)|(y1<0)|(max(w,h)<40)|(min(w,h)<=5)): 
                continue          
            num=40
            while(num):

                size=randint(np.floor(0.8*min(w,h)),np.ceil(1.25*max(w,h))+1)

                delta_w = randint(-w * 0.2, w * 0.2 + 1)
                delta_h = randint(-h * 0.2, h * 0.2 + 1)
                # random face box
                nx1 = int(max(x1 + w / 2 + delta_w - size / 2, 0))
                ny1 = int(max(y1 + h / 2 + delta_h - size / 2, 0))
                nx2 = nx1 + size
                ny2 = ny1 + size 

                if( nx2 > weight | ny2 > height):
                    continue               

                _box=[x1,y1,w,h]
                _box=np.array(_box).reshape(1,-1)     
                if(IoU(np.array([nx1,ny1,nx2,ny2]),_box)>0.65): 
                    facemark=np.zeros((5,2))
                    for i in range(5):
                        mark=((landmark[i][0]-nx1)/size,(landmark[i][1]-ny1)/size)
                        facemark[i]=mark  
                    img_list.append(cv2.resize(img[ny1:ny2,nx1:nx2,:], (img_size, img_size)))  
                    mark_list.append(facemark.reshape(10))

                    #mirro
                    mirro_mark=facemark.copy()
                    if(random.choice([0,1])):
                        img1,mirro_mark=flip(img[ny1:ny2,nx1:nx2,:],mirro_mark)
                        img_list.append(cv2.resize(img1, (img_size, img_size)))  
                        mark_list.append(mirro_mark.reshape(10))  

                    num=num-1
            for i in range(len(img_list)):

                if np.sum(np.where(mark_list[i] <= 0, 1, 0)) > 0:
                    continue

                if np.sum(np.where(mark_list[i] >= 1, 1, 0)) > 0:
                    continue

                cv2.imwrite(os.path.join(landmark_dir,'land_%d.jpg'%(land_idx)),img_list[i])
                mark=[str(_)for _ in mark_list[i]]
                f4.write(os.path.join(landmark_dir,'land_%d.jpg'%(land_idx)) +' -2 '+' '.join(mark)+'\n')
                land_idx=land_idx+1
            
        print("pics all done,land_pics %d in total"%(land_idx))
        
    f4.close()    

if __name__=="__main__":

    img_size=12
    
    #change img_size to P=12 R=24 O=48 net
    
    begin=time.time()
    
    base_dir="E:\\friedhelm\\object\\face_detection_MTCNN"
    
    lfw_dir=os.path.join(base_dir,"prepare_data")
    pic_spilt_dir=os.path.join(base_dir,"prepare_data\\trainImageList.txt")
    landmark_dir=os.path.join(base_dir,"DATA\\%d\\landmark"%(img_size))
    save_dir=os.path.join(base_dir,"DATA\\%d"%(img_size))

    if not os.path.exists(landmark_dir):
        os.makedirs(landmark_dir)    

    main()
    
    print(time.time()-begin)