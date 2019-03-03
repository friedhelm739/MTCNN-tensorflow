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
    
        
    def calibrate_box(self,img,NMS_box):
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
        net_box=[]
        score_box=[] 
        landmark_box=[]
        h,w,c=img.shape

        
        for box_ in NMS_box:

            box=box_.copy()                        

            if((box[0]<0)|(box[1]<0)|(box[2]>w)|(box[3]>h)|(box[2]-box[0]<self.min_face_size)|(box[3]-box[1]<self.min_face_size)): 
                continue  

            # calibrate the boundingbox

            t_box=[0]*4
            t_w=box[2]-box[0]+1
            t_h=box[3]-box[1]+1

            t_box[0]=box[5]*t_w+box[0]
            t_box[1]=box[6]*t_h+box[1]                     
            t_box[2]=box[7]*t_w+box[2]    
            t_box[3]=box[8]*t_h+box[3]                        
            
            if((t_box[0]<0)|(t_box[1]<0)|(t_box[2]>w)|(t_box[3]>h)|(t_box[2]-t_box[0]<self.min_face_size)|(t_box[3]-t_box[1]<self.min_face_size)): 
                continue 
            
            landmark=np.zeros((5,2))
            for i in range(5):
                landmark[i]=(box_[9+i*2]*(t_box[2]-t_box[0]+1)+t_box[0],box_[9+i*2+1]*(t_box[3]-t_box[1]+1)+t_box[1])
                
            landmark_box.append(landmark)            
            score_box.append(box_[4])
            net_box.append(t_box)     
            
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
            pnet_box  : list of box after calibration  -1*[p_face_x1,p_face_x2,p_face_y1,p_face_y2]
        """        
        factor=self.factor
        pro=12/self.min_face_size
        scales=[]
        total_box=[]
        score_box=[]
        small=min(img.shape[0:2])*pro

        while small>=12:
            scales.append(pro)
            pro*=factor
            small*=factor 
            
        for scale in scales:
            
            crop_img=img
            scale_img=cv2.resize(crop_img,((int(crop_img.shape[1]*scale)),(int(crop_img.shape[0]*scale))))
            scale_img1=np.reshape(scale_img,(-1,scale_img.shape[0],scale_img.shape[1],scale_img.shape[2])) 
            
            bounding_boxes=pnet_detector.predict(scale_img1,scale,img_size=12,stride=2,threshold=self.threshold[0],boxes=[])
            
            if(bounding_boxes):
                for box in bounding_boxes:
                    total_box.append(box)        

        NMS_box=NMS(total_box,0.7)                    
        if(len(NMS_box)==0):
            return None,None,None

        score_box,pnet_box,_=self.calibrate_box(img,NMS_box)
            
        return score_box,pnet_box,None
        

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
        score_box=[]        
        scale_img=np.zeros((len(bounding_box),24,24,3))
        
        for idx,box in enumerate(bounding_box):
            scale_img[idx,:,:,:] = cv2.resize(img[int(box[1]):int(box[3]),int(box[0]):int(box[2]),:], (24, 24))
            
        bounding_boxes=rnet_detector.predict(scale_img,scale=1,img_size=24,stride=4,threshold=self.threshold[1],boxes=bounding_box)
                    
        NMS_box=NMS(bounding_boxes,0.6)                    
                    
        if(len(NMS_box)==0):
            return None,None,None

        score_box,rnet_box,_=self.calibrate_box(img,NMS_box)
        
        return score_box,rnet_box,None        

    
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
        score_box=[] 
        landmark_box=[]
        
        scale_img=np.zeros((len(bounding_box),48,48,3))
        
        for idx,box in enumerate(bounding_box):
            scale_img[idx,:,:,:] = cv2.resize(img[int(box[1]):int(box[3]),int(box[0]):int(box[2]),:], (48, 48))
            
        bounding_boxes=onet_detector.predict(scale_img,scale=1,img_size=48,stride=8,threshold=self.threshold[2],boxes=bounding_box)
                    
        NMS_box=NMS(bounding_boxes,0.6)                    
                    
        if(len(NMS_box)==0):
            return None,None,None

        score_box,onet_box,landmark_box=self.calibrate_box(img,NMS_box)     

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
                pnet_detector=Detector(self.pnet_model,self.model_path[0],self.batch_size)
                score_box,bounding_box,landmark_box=self.detect_Pnet(pnet_detector,img)
                
                print("pnet-time: ",time.time()-pt)
                if((bounding_box is None) or (len(bounding_box)==0)):
                    face_boxes.append([])
                    landmark_boxes.append([])                    
                    continue

            if self.rnet_model:
                rt=time.time()
                batch_size=len(bounding_box)
                rnet_detector=Detector(self.rnet_model,self.model_path[1],batch_size)                 
                score_box,bounding_box,landmark_box=self.detect_Rnet(rnet_detector,img,bounding_box)
                
                print("rnet-time: ",time.time()-rt)
                if((bounding_box is None) or (len(bounding_box)==0)):
                    face_boxes.append([])
                    landmark_boxes.append([])                    
                    continue
                   
            if self.onet_model:
                ot=time.time()
                batch_size=len(bounding_box)
                onet_detector=Detector(self.onet_model,self.model_path[2],batch_size)                
                score_box,bounding_box,landmark_box=self.detect_Onet(onet_detector,img,bounding_box)

                print("onet-time: ",time.time()-ot)                
                if((bounding_box is None) or (len(bounding_box)==0)):
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
        
        if self.pnet_model:
            pt=time.time()
            pnet_detector=Detector(self.pnet_model,self.model_path[0],self.batch_size)
            score_box,bounding_box,landmark_box=self.detect_Pnet(pnet_detector,img)
            
            print("pnet-time: ",time.time()-pt)
            if((bounding_box is None) or (len(bounding_box)==0)):
                return [],[]

        if self.rnet_model:
            rt=time.time()
            batch_size=len(bounding_box)
            rnet_detector=Detector(self.rnet_model,self.model_path[1],batch_size)                 
            score_box,bounding_box,landmark_box=self.detect_Rnet(rnet_detector,img,bounding_box)
            
            print("rnet-time: ",time.time()-rt)
            if((bounding_box is None) or (len(bounding_box)==0)):
                return [],[]
            
        if self.onet_model:
            ot=time.time()
            batch_size=len(bounding_box)
            onet_detector=Detector(self.onet_model,self.model_path[2],batch_size)                
            score_box,bounding_box,landmark_box=self.detect_Onet(onet_detector,img,bounding_box)

            print("onet-time: ",time.time()-ot)            
            if((bounding_box is None) or (len(bounding_box)==0)):
                return [],[]
        
        print("detect-time: ",time.time()-detect_begin)
            
        return bounding_box,landmark_box