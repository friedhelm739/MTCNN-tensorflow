# -*- coding: utf-8 -*-
"""
@author: friedhelm

ops: too slow to generate pics and may have some problems in theory
"""
import tensorflow as tf 
import numpy as np
import cv2
import time
from core.tool import IoU,NMS
import os
from core.MTCNN_model import Rnet_model

begin=time.time()

def featuremap1(sess,graph,img,scales,map_shape,stride,threshold):
    
    crop_img=img
    boundingBox=[]
    
    for scale in scales:
    
        img=cv2.resize(crop_img,((int(crop_img.shape[1]*scale)),(int(crop_img.shape[0]*scale))))
        
        left=0
        up=0
        
        images=graph.get_tensor_by_name("input/image:0")
        label= graph.get_tensor_by_name("output/label:0")
        roi= graph.get_tensor_by_name("output/roi:0")
        landmark= graph.get_tensor_by_name("output/landmark:0")
        img1=np.reshape(img,(-1,img.shape[0],img.shape[1],img.shape[2]))
    
        a,b,c=sess.run([label,roi,landmark],feed_dict={images:img1})
        a=np.reshape(a,(-1,2)) 
        b=np.reshape(b,(-1,4)) 
        c=np.reshape(c,(-1,10)) 
        for idx,prob in enumerate(a):
                
            if prob[1]>threshold:
                biasBox=[]
                biasBox.extend([float(left*stride)/scale,float(up*stride)/scale, float(left*stride+map_shape)/scale, float(up*stride+map_shape)/scale,prob[1]])
                biasBox.extend(b[idx])
                biasBox.extend(c[idx])
                boundingBox.append(biasBox)
                
            #防止左越界与下越界
            if (left*stride+map_shape<img.shape[1]):
                left+=1
            elif (up*stride+map_shape<img.shape[0]): 
                left=0
                up+=1
            else : break
            
    return boundingBox


def featuremap2(sess,images,label,roi,landmark,img,scales,map_shape,stride,threshold,P_net_box):
    
    boundingBox=[]
    
    a,b,c=sess.run([label,roi,landmark],feed_dict={images:img})
    a=np.reshape(a,(-1,2)) 
    b=np.reshape(b,(-1,4)) 
    c=np.reshape(c,(-1,10)) 
    
    for idx,prob in enumerate(a):

        if prob[1]>threshold:
            biasBox=[]
            biasBox.extend([P_net_box[idx][0],P_net_box[idx][1],P_net_box[idx][2],P_net_box[idx][3],prob[1]])
            biasBox.extend(b[idx])
            biasBox.extend(c[idx])
            boundingBox.append(biasBox)
      
    return boundingBox


img_size=48
P_map_shape=12
R_map_shape=24
stride=2
factor=0.79
threshold=0.8

P_graph_path='E:\\friedhelm\\object\\face_detection_MTCNN\\\model\\Pnet_model.ckpt-60000.meta'
P_model_path='E:\\friedhelm\\object\\face_detection_MTCNN\\\model\\Pnet_model.ckpt-60000'

R_graph_path='E:\\friedhelm\\object\\face_detection_MTCNN\\\Rnet_model\\Rnet_model.ckpt-40000.meta'
R_model_path='E:\\friedhelm\\object\\face_detection_MTCNN\\\Rnet_model\\Rnet_model.ckpt-40000'


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
    
f1 = open(os.path.join(save_dir, 'pos_%d.txt'%(img_size)), 'w')
f2 = open(os.path.join(save_dir, 'neg_%d.txt'%(img_size)), 'w')
f3 = open(os.path.join(save_dir, 'par_%d.txt'%(img_size)), 'w')   

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
                if(idx<5000):
                    idx+=1
                    continue
                img=cv2.imread(os.path.join(WIDER_dir,pic_dir).replace('/','\\'))
                h,w,c=img.shape
                print(pic_dir)
                if(min(h,w)<20):
                    continue
                model_begin=time.time()
                scales=[]
                total_box=[]
                small=min(img.shape[0:2])
                pro=12/20
        
                while small>=20:
                    scales.append(pro)
                    pro*=factor
                    small*=factor 
                    
                #方式1
                P_graph=tf.Graph()
                with P_graph.as_default():
                    P_saver=tf.train.import_meta_graph(P_graph_path)
                    with tf.Session(graph=P_graph) as sess:
                        P_saver.restore(sess,P_model_path)
                        bounding_boxes=featuremap1(sess,P_graph,img,scales,P_map_shape,2,threshold)
                            
                
                P_NMS_box=NMS(bounding_boxes,0.7)
                P_net_box=[]
                for box_ in P_NMS_box:

                    box=box_.copy()                        
                    if((box[0]<0)|(box[1]<0)|(box[2]>w)|(box[3]>h)|(box[2]-box[0]<=20)|(box[3]-box[1]<=20)): 
                        continue  
                    # format of total_box: [x1,y1,x2,y2,score,offset_x1,offset_y1,offset_x2,offset_y2,10*landmark]  

                    t_box=[0]*4
                    t_w=box[2]-box[0]+1
                    t_h=box[3]-box[1]+1

                    t_box[0]=box[5]*t_w+box[0]
                    t_box[1]=box[6]*t_h+box[1]                     
                    t_box[2]=box[7]*t_w+box[2]    
                    t_box[3]=box[8]*t_h+box[3]                        
                    # 计算真实人脸框

                    if((t_box[0]<0)|(t_box[1]<0)|(t_box[2]>img.shape[1])|(t_box[3]>img.shape[0])|(t_box[2]-t_box[0]<=20)|(t_box[3]-t_box[1]<=20)): 
                        continue 

                    ti_box=t_box.copy()
                    ti_box=[int(_) for _ in ti_box]                        

                    P_net_box.append(ti_box)
                
                R_graph=tf.Graph()
                with R_graph.as_default():
                    images = tf.placeholder(tf.float32)
                    label,roi,landmark=Rnet_model(images,len(P_net_box))                    
#                     R_saver=tf.train.import_meta_graph(R_graph_path) 
                    R_saver=tf.train.Saver()
                    with tf.Session() as sess:      
                        R_saver.restore(sess,R_model_path)
                        resized_img=np.zeros((len(P_net_box),R_map_shape,R_map_shape,3),dtype=np.float32)
                        for i,box in enumerate(P_net_box):
                            resized_img[i,:,:,:] = cv2.resize(img[box[1]:box[3],box[0]:box[2],:], (R_map_shape,R_map_shape))
                        bounding_boxes=featuremap2(sess,images,label,roi,landmark,resized_img,1,R_map_shape,4,0.85,P_net_box)                   
              
                R_NMS_box=NMS(bounding_boxes,0.6)                                       
                neg_num=0
                
                for box_ in R_NMS_box:
                
                    box=box_.copy()                        
                    if((box[0]<0)|(box[1]<0)|(box[2]>w)|(box[3]>h)|(box[2]-box[0]<=20)|(box[3]-box[1]<=20)): 
                        continue  
                    # format of total_box: [x1,y1,x2,y2,score,offset_x1,offset_y1,offset_x2,offset_y2,10*landmark]  

                    t_box=[0]*4
                    t_w=box[2]-box[0]+1
                    t_h=box[3]-box[1]+1

                    t_box[0]=box[5]*t_w+box[0]
                    t_box[1]=box[6]*t_h+box[1]                     
                    t_box[2]=box[7]*t_w+box[2]    
                    t_box[3]=box[8]*t_h+box[3]                        
                    # 计算真实人脸框

                    if((t_box[0]<0)|(t_box[1]<0)|(t_box[2]>img.shape[1])|(t_box[3]>img.shape[0])|(t_box[2]-t_box[0]<=20)|(t_box[3]-t_box[1]<=20)): 
                        continue 

                    ti_box=t_box.copy()
                    ti_box=[int(_) for _ in ti_box]

                    Iou=IoU(np.array(t_box),np.array(boxes))

                    if(np.max(Iou)<0.3):
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
                print(time.time()-model_begin)
                idx+=1
                if(idx%100==0):
                    print('idx: ',idx," ;neg_idx: ",neg_idx," ;pos_idx: ",pos_idx," ;par_idx: ",par_idx)
                    print(time.time()-begin)
    print("pics all done,neg_pics %d in total,pos_pics %d in total,par_pics %d in total"%(neg_idx,pos_idx,par_idx))
    
f1.close()
f2.close()
f3.close()  
print(time.time()-begin )
# idx:  10000  ;neg_idx:  9825  ;pos_idx:  98071  ;par_idx:  32023