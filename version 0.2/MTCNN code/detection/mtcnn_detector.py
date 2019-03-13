# -*- coding: utf-8 -*-
"""
@author: friedhelm

"""
import numpy as np
import cv2
from core.tool import NMS
from detection.detector import Detector
import time

class MTCNN_Detector(object):
    
    def __init__(self,model,model_path,batch_size,factor,min_face_size,threshold):
        
        self.pnet_model=model[0]
        self.rnet_model=model[1]       
        self.onet_model=model[2]        
        self.model_path=model_path        
        self.batch_size=batch_size        
        self.factor=factor
        self.min_face_size=min_face_size 
        self.threshold=threshold
        self.pnet_detector=Detector(self.pnet_model,self.model_path[0],"Pnet",self.batch_size)
        self.rnet_detector=Detector(self.rnet_model,self.model_path[1],"Rnet",-1)        
        self.onet_detector=Detector(self.onet_model,self.model_path[2],"Onet",-1)
        
        
    def generate_box(self,score_box,bounding_box,img_size,scale,stride,threshold):
        
        idx=np.where(score_box>threshold)
    
        if(idx[0].size==0):
            return np.array([])

        delta_x1,delta_y1,delta_x2,delta_y2=[bounding_box[idx[0],idx[1],i] for i in range(4)]
        delta_box=np.array([delta_x1,delta_y1,delta_x2,delta_y2])
        score=score_box[idx[0],idx[1]]        
        
        return np.vstack([(idx[1]*stride/scale),
                          (idx[0]*stride/scale),
                          ((idx[1]*stride+img_size)/scale),
                          ((idx[0]*stride+img_size)/scale),
                          score,
                          delta_box]).T
    
    
    def pad(self,t_box,w,h):
        """
        used for pad t_box that out of range
        
        input : t_box; wide,high of img
        output: boundingbox after pad
        
        format of input  : 
            t_box : [x1,y1,x2,y2] 
            w       : int
            h       : int
            
        format of output : 
            t_box    : [x1,y1,x2,y2]
        """  
        xl_idx=np.where((t_box[:,0]<0))[0]  
        t_box[xl_idx]=0
        
        yl_idx=np.where((t_box[:,1]<0))[0]  
        t_box[yl_idx]=0  
        
        xr_idx=np.where((t_box[:,2]>w))[0]  
        t_box[xr_idx]=w-1

        yr_idx=np.where((t_box[:,3]>h))[0]  
        t_box[yr_idx]=h-1  
        
        return t_box
    
    def convert_to_square(self,box):
        """
        used for convert box to square
        
        input : boundingbox 
        output: boundingbox
        
        format of input  : 
            box : -1*[x1,y1,x2,y2] 
            
        format of output : 
            square_box : -1*[x1,y1,x2,y2]
        """
        square_box=box.copy()
        
        h=box[:,3]-box[:,1]+1
        w=box[:,2]-box[:,0]+1

        max_side=np.maximum(w,h)
        
        square_box[:,0]=box[:,0]+w*0.5-max_side*0.5
        square_box[:,1]=box[:,1]+h*0.5-max_side*0.5
        square_box[:,2]=square_box[:,0]+max_side-1
        square_box[:,3]=square_box[:,1]+max_side-1
        
        return square_box

    
    def calibrate_box(self,img,NMS_box,model_name="default"):
        """
        used for calibrating NMS_box
        
        input : boundingbox after nms, img
        output: score , boundingbox after calibrate , landmark_box
        
        format of input  : 
            NMS_box : -1*[x1,y1,x2,y2,score,offset_x1,offset_y1,offset_x2,offset_y2,5*(landmark_x,landmark_y)] 
            img          : np.array()
            
        format of output : 
            score_box    : list of score -1*[score]
            net_box      : list of box   -1*[face_x1,face_x2,face_y1,face_y2]
            landmark_box : list of box   -1*[5*(true_landmark_x,true_landmark_y)]
        """
        landmark_box=np.array([])
        h,w,c=img.shape
        
        t_box=np.zeros((NMS_box.shape[0],4))
        boxes=np.vstack(NMS_box)
        
        bounding_box=boxes[:,0:4]
        delta_box=boxes[:,5:9]
        
        t_w=bounding_box[:,2]-bounding_box[:,0]+1
        t_w=np.expand_dims(t_w,1)
        t_h=bounding_box[:,3]-bounding_box[:,1]+1
        t_h=np.expand_dims(t_h,1)
        
        w_h=np.hstack([t_w,t_h,t_w,t_h])
        t_box=bounding_box+delta_box*w_h
        
        if(model_name!="Onet"):
            t_box=self.convert_to_square(t_box)
            
        t_box=self.pad(t_box,w,h)
        idx=np.where((t_box[:,2]-t_box[:,0]>=self.min_face_size)&(t_box[:,3]-t_box[:,1]>=self.min_face_size))[0]
        net_box=t_box[idx]
       
        if(model_name=="Onet"):
            boxes=boxes[idx]
            score_box=boxes[:,4]
            landmark_box=np.zeros((len(boxes),5,2))
            for i in range(5):
                landmark_box[:,i,:]=(boxes[:,9+i*2]*(t_box[:,2]-t_box[:,0])+t_box[:,0],boxes[:,9+i*2+1]*(t_box[:,3]-t_box[:,1])+t_box[:,1])   
        else: 
            score_box=boxes[:,4][idx]
        
        return score_box,net_box,landmark_box
   

    def detect_Pnet(self,pnet_detector,img):
        """
        input : detector , img
        output: score_box , pnet_box , None
        
        format of input  :
            detector: class detector 
            img     : np.array()
            
        format of output : 
            score_box : list of score                  -1*[score]
            pnet_box  : np.array() of box after calibration  -1*[p_face_x1,p_face_x2,p_face_y1,p_face_y2]
        """        
        factor=self.factor
        pro=12/self.min_face_size
        scales=[]
        total_box=[]
        small=min(img.shape[0:2])*pro

        while small>=12:
            scales.append(pro)
            pro*=factor
            small*=factor 
            
        for scale in scales:
            
            crop_img=img
            scale_img=cv2.resize(crop_img,((int(crop_img.shape[1]*scale)),(int(crop_img.shape[0]*scale))))
            scale_img1=np.reshape(scale_img,(-1,scale_img.shape[0],scale_img.shape[1],scale_img.shape[2])) 
            
            score_boxes,delta_boxes,_=pnet_detector.predict(scale_img1)
            
            bounding_boxes=self.generate_box(score_box=score_boxes[:,:,1],bounding_box=delta_boxes,img_size=12,scale=scale,stride=2,threshold=self.threshold[0])

            if(len(bounding_boxes)!=0):
                idx=NMS(bounding_boxes[:,0:5],0.5)
                NMS_bounding_boxes=bounding_boxes[idx]
                total_box.append(NMS_bounding_boxes) 

        total_box=np.vstack(total_box)
        idx=NMS(total_box,0.7)
        NMS_box=total_box[idx]       
        if(len(NMS_box)==0):
            return [],[],[]

        score_box,pnet_box,_=self.calibrate_box(img,NMS_box)
            
        return score_box,pnet_box,[]
        

    def detect_Rnet(self,rnet_detector,img,bounding_box):
        """
        input : detector , img , bounding_box
        output: score_box , rnet_box , None
        
        format of input  :
            detector     : class detector 
            img          : np.array()
            bounding_box : list of box output from function(detect_Pnet)  -1*[p_face_x1,p_face_x2,p_face_y1,p_face_y2]
            
        format of output : 
            score_box : list of score                  -1*[score]
            rnet_box  : list of box after calibration  -1*[r_face_x1,r_face_x2,r_face_y1,r_face_y2]
        """               
        scale_img=np.zeros((len(bounding_box),24,24,3))
        for idx,box in enumerate(bounding_box):
            scale_img[idx,:,:,:] = cv2.resize(img[int(box[1]):int(box[3]),int(box[0]):int(box[2]),:], (24, 24))
            
        score_boxes,delta_boxes,_=rnet_detector.predict(scale_img)
        
        idx=np.where(score_boxes[:,1]>self.threshold[1])[0] 
        
        if(len(idx)==0):
            return [],[],[]        
        delta_boxes=delta_boxes[idx]
        bounding_box=bounding_box[idx]
        score_boxes=np.expand_dims(score_boxes[idx,1],1)
        bounding_box=np.hstack([bounding_box,score_boxes])
        
        idx=NMS(bounding_box,0.6)                    
        bounding_box=bounding_box[idx]  
        delta_boxes=delta_boxes[idx]
        
        NMS_box=np.hstack([bounding_box,delta_boxes])
          
        score_box,rnet_box,_=self.calibrate_box(img,NMS_box)
        
        return score_box,rnet_box,[]        

    
    def detect_Onet(self,onet_detector,img,bounding_box):
        """
        input : detector , img , bounding_box
        output: score_box , onet_box , landmark_box
        
        format of input  :
            detector     : class detector 
            img          : np.array()
            bounding_box : list of box output from function(detect_Rnet)  -1*[r_face_x1,r_face_x2,r_face_y1,r_face_y2]
            
        format of output : 
            score_box    : list of score                  -1*[score]
            onet_box     : list of box after calibration  -1*[o_face_x1,o_face_x2,o_face_y1,o_face_y2]
            landmark_box : list of landmark               -1*[5*(o_landmark_x,o_landmark_y)]
        """              
        scale_img=np.zeros((len(bounding_box),48,48,3))    
        for idx,box in enumerate(bounding_box):
            scale_img[idx,:,:,:] = cv2.resize(img[int(box[1]):int(box[3]),int(box[0]):int(box[2]),:], (48, 48))
            
        score_boxes,delta_boxes,landmark_boxes=onet_detector.predict(scale_imgthreshold=self.threshold[2])
        
        idx=np.where(score_boxes[:,1]>self.threshold[1])[0] 
        
        if(len(idx)==0):
            return [],[],[]        
        delta_boxes=delta_boxes[idx]
        bounding_box=bounding_box[idx]
        bounding_box[:,4]=score_boxes[idx,1]
        landmark_boxes=landmark_boxes[idx]
        
        idx=NMS(bounding_box,0.6)                    
        bounding_box=bounding_box[idx]  
        delta_boxes=delta_boxes[idx]       
        landmark_boxes=landmark_boxes[idx]        
        
        NMS_box=np.hstack([bounding_box,delta_boxes,landmark_boxes])
        
        score_box,onet_box,landmark_box=self.calibrate_box(img,NMS_box,"Onet")     

        return score_box,onet_box,landmark_box   
    
    
    def detect_face(self,images):    
        """
        used for detecting face in both batch images and single image
        
        input : images 
        output: face_boxes , landmark_boxes
        
        format of input  :
            img          : np.array() batch_size*single_img
            
        format of output : 
            face_boxes     : list of face_box      batch_size*[face_x1,face_x2,face_y1,face_y2]
            landmark_boxes : list of landmark_box  batch_size*[5*(landmark_x,landmark_y)]
        """
        sign=False 
        bounding_box=[]
        landmark_box=[]
        face_boxes=[]
        landmark_boxes=[]
        detect_begin=time.time()
        
        if(np.size(images.shape)==3):
            sign=True
            img=np.zeros((1,images.shape[0],images.shape[1],images.shape[2]))
            img[0,:,:,:]=images
            images=img 
            
        for img in images:

            if(img is None):
                face_boxes.append([])
                landmark_boxes.append([])     
                continue
            
            if self.pnet_model:
                pt=time.time()
                score_box,bounding_box,landmark_box=self.detect_Pnet(self.pnet_detector,img)
                
                print("pnet-time: ",time.time()-pt)
                if(len(bounding_box)==0):
                    face_boxes.append([])
                    landmark_boxes.append([])                    
                    continue

            if self.rnet_model:
                rt=time.time()
                score_box,bounding_box,landmark_box=self.detect_Rnet(self.rnet_detector,img,bounding_box)
                
                print("rnet-time: ",time.time()-rt)
                if(len(bounding_box)==0):
                    face_boxes.append([])
                    landmark_boxes.append([])                    
                    continue
                   
            if self.onet_model:
                ot=time.time()
                score_box,bounding_box,landmark_box=self.detect_Onet(self.onet_detector,img,bounding_box)

                print("onet-time: ",time.time()-ot)                
                if(len(bounding_box)==0):
                    face_boxes.append([])
                    landmark_boxes.append([])                    
                    continue

            face_boxes.append(bounding_box)
            landmark_boxes.append(landmark_box)
        
        print("detect-time: ",time.time()-detect_begin)
        if(sign):
            return face_boxes[0],landmark_boxes[0]
        else:
            return face_boxes,landmark_boxes
    
    
    def detect_single_face(self,img):    
        """
        used for detecting single face or vidio
        
        input : images 
        output: bounding_box , landmark_box
        
        format of input  :
            img          : np.array() 
            
        format of output : 
            bounding_box : list of box  [face_x1,face_x2,face_y1,face_y2]
            landmark_box : list of box  [5*(landmark_x,landmark_y)]
        """    
        bounding_box=[]
        landmark_box=[]     
        detect_begin=time.time()
        
        if(img is None):
            return [],[]            
        
        img=(img-127.5)/128
        
        if self.pnet_model:
            score_box,bounding_box,landmark_box=self.detect_Pnet(self.pnet_detector,img)
            
            if(len(bounding_box)==0):
                return [],[]

        if self.rnet_model:              
            score_box,bounding_box,landmark_box=self.detect_Rnet(self.rnet_detector,img,bounding_box)
            
            if(len(bounding_box)==0):
                return [],[]
            
        if self.onet_model:           
            score_box,bounding_box,landmark_box=self.detect_Onet(self.onet_detector,img,bounding_box)
          
            if(len(bounding_box)==0):
                return [],[]
        
        print("detect-time: ",time.time()-detect_begin)
            
        return bounding_box,landmark_box