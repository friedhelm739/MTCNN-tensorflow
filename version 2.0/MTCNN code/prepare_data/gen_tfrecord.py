# -*- coding: utf-8 -*-
"""
@author: friedhelm

"""
import tensorflow as tf 
import cv2
import random
import time
import os
import argparse

def arg_parse():
    
    parser=argparse.ArgumentParser()
    
    parser.add_argument("--img_size",default=12 , type=int, help='img size to generate')
    parser.add_argument("--base_dir",default="../" , type=str, help='base path to save TFRecord file')
    parser.add_argument("--base_num",default=200000 , type=int, help='base num img  to generate')
    
    return parser


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
    
    parser=arg_parse()
    base_dir=parser.base_dir
    img_size=parser.img_size
    base=parser.base_num

    terms=['neg_%d'%(img_size),'pos_%d'%(img_size),'par_%d'%(img_size),'land_%d'%(img_size)]
    scale=[3,1,1,2]
    
    #set base number of pos_pic    
    #base=200000    
    #img_size=12    
    #base_dir="/home/dell/Desktop/MTCNN"

    begin=time.time()

    main()
    
    print(time.time()-begin)