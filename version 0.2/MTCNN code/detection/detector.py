# -*- coding: utf-8 -*-
"""
@author: friedhelm

"""
import tensorflow as tf
import numpy as np
import os

class Detector(object):
    
    def __init__(self,model,model_path,model_name,batch_size):
        
        if(model_path):
            path=model_path.replace("/","\\")
            model_name=path.split("\\")[-1].split(".")[0]
            if not os.path.exists(model_path+".meta"):
                raise Exception("%s is not exists"%(model_name))
                
            graph=tf.Graph()
            with graph.as_default():
                self.sess=tf.Session()
                self.images=tf.placeholder(tf.float32)
                self.label,self.roi,self.landmark=model(self.images,batch_size) 
                saver=tf.train.Saver()
                saver.restore(self.sess,model_path)
            
            self.model_name=model_name
        
    def predict(self,img):
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
        pre_land=np.array([0])
        if(self.model_namemodel_name=="Onet"):
            pre_label,pre_box,pre_land=self.sess.run([self.label,self.roi,self.landmark],feed_dict={self.images:img})            
        else:
            pre_label,pre_box=self.sess.run([self.label,self.roi],feed_dict={self.images:img}) 
            
        return np.vstack(pre_label),np.vstack(pre_box),np.vstack(pre_land)