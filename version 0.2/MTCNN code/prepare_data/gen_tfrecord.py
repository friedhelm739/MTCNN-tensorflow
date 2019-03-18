# -*- coding: utf-8 -*-
"""
@author: friedhelm

"""
import tensorflow as tf 
import cv2
import random
import time
import os

def main():
    
    t_time=time.time()   
    for index,term in enumerate(terms):
        num=0
        print("%s start"%(term))
        tfr_addr=os.path.join(base_dir,"DATA/%d/%s_train.tfrecords"%(img_size,term))
        with tf.python_io.TFRecordWriter(tfr_addr) as writer:
            file_addr=os.path.join(base_dir,"DATA/%d/%s.txt"%(img_size,term))
            with open(file_addr) as readlines:
                readlines=[line.strip().split(' ') for line in readlines]
                random.shuffle(readlines)
                for i,line in enumerate(readlines):
                    if(num%10000==0):
                        print(i,time.time()-t_time)
                        t_time=time.time()
                    img=cv2.imread(line[0])
                    if(img is None):
                        continue
                    img_raw = img.tobytes()
                    label=int(line[1])
                    roi=[0.0]*4               
                    landmark=[0.0]*10
                    if(len(line)==6):    
                        roi=[float(_) for _ in line[2:6]]   
                    if(len(line)==12):
                        landmark=[float(_) for _ in line[2:12]]                  
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                        "roi": tf.train.Feature(float_list=tf.train.FloatList(value=roi)),
                        "landmark": tf.train.Feature(float_list=tf.train.FloatList(value=landmark)),
                    }))
                    writer.write(example.SerializeToString())  #序列化为字符串  
                    num+=1
                    if(num==base*scale[index]):
                        print("%s finish"%(term))
                        break

if __name__=="__main__":
    
    img_size=12
    #change img_size to P=12 R=24 O=48 net
    terms=['neg_%d'%(img_size),'pos_%d'%(img_size),'par_%d'%(img_size),'land_%d'%(img_size)]
    scale=[3,1,1,2]
    
    base_dir="/home/dell/Desktop/prepared_data"

    #set base number of pos_pic    
    base=200000

    begin=time.time()

    main()
    
    print(time.time()-begin)