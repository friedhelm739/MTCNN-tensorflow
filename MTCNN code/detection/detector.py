# -*- coding: utf-8 -*-
"""
@author: friedhelm

"""
import tensorflow as tf
import numpy as np
import os

class Detector(object):
    
    def __init__(self,model,model_path,batch_size):
        
        model_name=model_path.split("\\")[-1].split(".")[0]
        if not os.path.exists(model_path+".meta"):
            raise Exception("%s is not exists"%(model_name))
            
        graph=tf.Graph()
        with graph.as_default():
            self.sess=tf.Session()
            self.images=tf.placeholder(tf.float32)
            self.label,self.roi,self.landmark=model(self.images,batch_size) 
            saver=tf.train.Saver()
            saver.restore(self.sess,model_path)

    def predict(self,img,scale,img_size,stride,threshold,boxes):
        """
        used for predict
        
        input : img, scale , img_size , stride , threshold , boxes
        output: boundingbox
        
        format of input  : 
            img       : np.array()
            scale     : float , which img will be resizeed to
            img_size  : int   , size of img
            stride    : int   , stride of the featrue map of model
            threshold : int   , percentage of result to keep
            boxes     : list  , output bounding box of pre-model
        format of output : 
            boundingbox : list of boxes -1*[x1,y1,x2,y2,score,offset_x1,offset_y1,offset_x2,offset_y2,5*(landmark_x,landmark_y)]
        """    
        left=0
        up=0
        boundingBox=[]

        pre_label,pre_box,pre_land=self.sess.run([self.label,self.roi,self.landmark],feed_dict={self.images:img})
        
        pre_label=np.reshape(pre_label,(-1,2)) 
        pre_box=np.reshape(pre_box,(-1,4)) 
        pre_land=np.reshape(pre_land,(-1,10))

        for idx,prob in enumerate(pre_label):

            if prob[1]>threshold:
                biasBox=[]
                if(len(boxes) == 0):
                    biasBox.extend([float(left*stride)/scale,float(up*stride)/scale, float(left*stride+12)/scale, float(up*stride+12)/scale,prob[1]])
                else:
                    biasBox.extend([boxes[idx][0],boxes[idx][1],boxes[idx][2],boxes[idx][3],prob[1]])
                biasBox.extend(pre_box[idx])
                biasBox.extend(pre_land[idx])
                boundingBox.append(biasBox)

            if (len(boxes) == 0):
                #prevent the sliding window to overstep the boundary
                if (left*stride+img_size<img.shape[2]):
                    left+=1
                elif (up*stride+img_size<img.shape[1]): 
                    left=0
                    up+=1
                else : continue

        return boundingBox    