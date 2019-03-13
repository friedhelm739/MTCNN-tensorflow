# -*- coding: utf-8 -*-
"""
@author: friedhelm

"""
import tensorflow as tf

def label_los(pre_label,act_label):
    
    ratio=tf.constant(0.7)
    zeros=tf.zeros_like(act_label,dtype=tf.int32)
    valid_label=tf.where(tf.less(act_label,0),zeros,act_label)

    column_num=tf.shape(pre_label,out_type=tf.int32)[0]
    pre_label=tf.squeeze(tf.reshape(pre_label,(1,-1)))
    column=tf.range(0,column_num)*2 
    column_to_stay=column+valid_label

    pre_label=tf.squeeze(tf.gather(pre_label,column_to_stay))
    loss = -tf.log(pre_label+1e-10)      
    ones=tf.ones_like(act_label,dtype=tf.float32)
    zero=tf.zeros_like(act_label,dtype=tf.float32)
    valid_colunm = tf.where(act_label < zeros,zero,ones)  
    
    num_column=tf.reduce_sum(valid_colunm)
    num=tf.cast(num_column*ratio,dtype=tf.int32)
    loss=tf.multiply(loss,valid_colunm,'label_los')
    loss,_=tf.nn.top_k(loss,num)
    
    return tf.reduce_mean(loss)
    

def roi_los(label,pre_box,act_box) :    
    
    zeros=tf.zeros_like(label,dtype=tf.float32)
    ones=tf.ones_like(label,dtype=tf.float32)    
    valid_label=tf.where(tf.equal(abs(label),1),ones,zeros)
    loss=tf.reduce_sum(tf.square(act_box-pre_box),axis=1)
    loss=tf.multiply(loss,valid_label,'roi_los')
    return tf.reduce_mean(loss) 
    

def landmark_los(label,pre_landmark,act_landmark):    
    
    zeros=tf.zeros_like(label,dtype=tf.float32)
    ones = tf.ones_like(label,dtype=tf.float32)
    valid_label=tf.where(tf.equal(label,-2),ones,zeros)
    loss=tf.reduce_sum(tf.square(act_landmark-pre_landmark),axis=1)
    loss=tf.multiply(loss,valid_label,'landmark_los')
    return tf.reduce_mean(loss)     
    

def cal_accuracy(cls_prob,label):
       
    pred = tf.argmax(cls_prob,axis=1)
    label_int = tf.cast(label,tf.int64)
    cond = tf.where(tf.greater_equal(label_int,0))
    picked = tf.squeeze(cond)
    label_picked = tf.gather(label_int,picked)
    pred_picked = tf.gather(pred,picked)
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(label_picked,pred_picked),tf.float32))
    
    return accuracy_op