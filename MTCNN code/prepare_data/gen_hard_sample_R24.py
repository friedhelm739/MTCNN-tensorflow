# -*- coding: utf-8 -*-
"""
@author: friedhelm

"""
#159424张人脸
import tensorflow as tf 
import numpy as np
import cv2
import time
from core.tool import IoU,NMS,featuremap
import os

def main():
    saver=tf.train.import_meta_graph(graph_path)  
    
    f1 = open(os.path.join(save_dir, 'pos_%d.txt'%(img_size)), 'w')
    f2 = open(os.path.join(save_dir, 'neg_%d.txt'%(img_size)), 'w')
    f3 = open(os.path.join(save_dir, 'par_%d.txt'%(img_size)), 'w')   

    with tf.Session() as sess: 

        saver.restore(sess,model_path)
        graph = tf.get_default_graph()

        with open(WIDER_spilt_dir) as filenames:
            p=0
            idx=0
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
                        print(pic_dir)
                        if(min(h,w)<20):
                            continue

                        scales=[]
                        total_box=[]
                        pro=map_shape/min_face_size
                        small=min(img.shape[0:2])*pro
                        

                        while small>=12:
                            scales.append(pro)
                            pro*=factor
                            small*=factor  

                        for scale in scales:

                            scale_img=cv2.resize(img,((int(img.shape[1]*scale)),(int(img.shape[0]*scale))))
                            bounding_boxes=featuremap(sess,graph,scale_img,scale,map_shape,stride,threshold)

                            if(bounding_boxes):
                                for box in bounding_boxes:
                                    total_box.append(box)

                        NMS_box=NMS(total_box,0.7)
                        neg_num=0
                        for box_ in NMS_box:

                            box=box_.copy()                        
                            if((box[0]<0)|(box[1]<0)|(box[2]>w)|(box[3]>h)|(box[2]-box[0]<=min_face_size)|(box[3]-box[1]<=min_face_size)): 
                                continue  
                            # format of total_box: [x1,y1,x2,y2,score,offset_x1,offset_y1,offset_x2,offset_y2,10*landmark]  

                            t_box=[0]*4
                            t_w=box[2]-box[0]+1
                            t_h=box[3]-box[1]+1

                            t_box[0]=box[5]*t_w+box[0]
                            t_box[1]=box[6]*t_h+box[1]                     
                            t_box[2]=box[7]*t_w+box[2]    
                            t_box[3]=box[8]*t_h+box[3]                        
                            # calculate ground truth predict-face boxes

                            if((t_box[0]<0)|(t_box[1]<0)|(t_box[2]>w)|(t_box[3]>h)|(t_box[2]-t_box[0]<=min_face_size)|(t_box[3]-t_box[1]<=min_face_size)): 
                                continue 

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

    f1.close()
    f2.close()
    f3.close()  


if __name__=="__main__":
    
    begin=time.time()
    img_size=24
    map_shape=12
    min_face_size=20
    stride=2
    factor=0.79
    threshold=0.8

    graph_path='E:\\friedhelm\\object\\face_detection_MTCNN\\\Pnet_model\\Pnet_model.ckpt-60000.meta'
    model_path='E:\\friedhelm\\object\\face_detection_MTCNN\\\Pnet_model\\Pnet_model.ckpt-60000'
    
    WIDER_dir="E:\\friedhelm\\object\\face_detection_MTCNN\\prepare_data\\WIDER_train\\images"
    WIDER_spilt_dir="E:\\friedhelm\\object\\face_detection_MTCNN\\prepare_data\\wider_face_split\\wider_face_train_bbx_gt.txt"
    negative_dir="E:\\friedhelm\\object\\face_detection_MTCNN\\DATA\\%d\\negative"%(img_size)
    positive_dir="E:\\friedhelm\\object\\face_detection_MTCNN\\DATA\\%d\\positive"%(img_size)
    par_dir="E:\\friedhelm\\object\\face_detection_MTCNN\\DATA\\%d\\part"%(img_size)
    save_dir="E:\\friedhelm\\object\\face_detection_MTCNN\\DATA\\%d"%(img_size)

    if not os.path.exists(positive_dir):
        os.makedirs(positive_dir)
    if not os.path.exists(par_dir):
        os.makedirs(par_dir)
    if not os.path.exists(negative_dir):
        os.makedirs(negative_dir)    
    
    main()
    
    print(time.time()-begin)    
    
#pics all done,neg_pics 758503 in total,pos_pics 285017 in total,par_pics 572771 in total
# 17590.795156002045