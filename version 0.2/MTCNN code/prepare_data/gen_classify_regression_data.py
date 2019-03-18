# -*- coding: utf-8 -*-
"""
@author: friedhelm

"""
import sys
sys.path.append("../")

import tensorflow as tf
import time
from core.tool import read_multi_tfrecords,image_color_distort
from core.MTCNN_model import Pnet_model,Rnet_model,Onet_model
from train.train_tool import label_los,roi_los,landmark_los,cal_accuracy
import os

def train(image,label,roi,landmark,model,model_name):
    
    _label, _roi ,_landmark=model(image,batch)
    
    with tf.name_scope('output'):
        _label=tf.squeeze(_label,name='label')
        _roi=tf.squeeze(_roi,name='roi')
        _landmark=tf.squeeze(_landmark,name='landmark')
        
    _label_los=label_los(_label,label)
    _box_los=roi_los(label,_roi,roi)    
    _landmark_los=landmark_los(label,_landmark,landmark)
    
    function_loss=_label_los*ratio[0]+_box_los*ratio[1]+_landmark_los*ratio[2]

    tf.add_to_collection("loss", function_loss)
    loss_all=tf.get_collection('loss')
    
    with tf.name_scope('loss'):
        loss=tf.reduce_sum(loss_all)
        tf.summary.scalar('loss',loss) 

    opt=tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    with tf.name_scope('accuracy'):
        train_accuracy,truth_accuracy,false_accuracy=cal_accuracy(_label,label)
        tf.summary.scalar('accuracy',train_accuracy) 
        tf.summary.scalar('ture_accuracy',truth_accuracy) 
        tf.summary.scalar('false_accuracy',false_accuracy) 
        
    saver=tf.train.Saver(max_to_keep=10)
    merged=tf.summary.merge_all() 
    
    images,labels,rois,landmarks=read_multi_tfrecords(addr,batch_size,img_size)   
    images=image_color_distort(images)
    
    with tf.Session() as sess:
        sess.run((tf.global_variables_initializer(),
                  tf.local_variables_initializer()))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        image_batch,label_batch,roi_batch,landmark_batch=sess.run([images,labels,rois,landmarks])
        
        writer_train=tf.summary.FileWriter(os.path.join(base_dir,"model/%s/"%(model_name)),sess.graph)
        try:
            
            for i in range(1,train_step):
                
                image_batch,label_batch,roi_batch,landmark_batch=sess.run([images,labels,rois,landmarks])
                
                sess.run(opt,feed_dict={image:image_batch,label:label_batch,roi:roi_batch,landmark:landmark_batch})
                if(i%100==0):
                    summary=sess.run(merged,feed_dict={image:image_batch,label:label_batch,roi:roi_batch,landmark:landmark_batch})
                    writer_train.add_summary(summary,i) 
                if(i%1000==0):
                    print('次数',i)    
                    print('train_accuracy',sess.run(train_accuracy,feed_dict={image:image_batch,label:label_batch,roi:roi_batch,landmark:landmark_batch}))
                    print('truth_accuracy',sess.run(truth_accuracy,feed_dict={image:image_batch,label:label_batch,roi:roi_batch,landmark:landmark_batch}))
                    print('false_accuracy',sess.run(false_accuracy,feed_dict={image:image_batch,label:label_batch,roi:roi_batch,landmark:landmark_batch}))
                    print('loss',sess.run(loss,{image:image_batch,label:label_batch,roi:roi_batch,landmark:landmark_batch})) 
                    print('_label_los',sess.run(_label_los,{image:image_batch,label:label_batch,roi:roi_batch,landmark:landmark_batch}))                     
                    print('_box_los',sess.run(_box_los,{image:image_batch,label:label_batch,roi:roi_batch,landmark:landmark_batch}))                     
                    print('_landmark_los',sess.run(_landmark_los,{image:image_batch,label:label_batch,roi:roi_batch,landmark:landmark_batch}))                     
                    print('time',time.time()-begin)
                    if(i%10000==0):
                        saver.save(sess,os.path.join(base_dir,"model/%s/%s.ckpt"%(model_name,model_name)),global_step=i)
        except  tf.errors.OutOfRangeError:
            print("finished")
        finally:
            coord.request_stop()
            writer_train.close()
        coord.join(threads)
    
def main(model):
    
    with tf.name_scope('input'):
        image=tf.placeholder(tf.float32,[batch,img_size,img_size,3],name='image')
        label=tf.placeholder(tf.int32,[batch],name='label')
        roi=tf.placeholder(tf.float32,[batch,4],name='roi')
        landmark = tf.placeholder(tf.float32,[batch,10],name='landmark')  

    train(image,label,roi,landmark,model,model_name)

if __name__=='__main__':
    
    base_dir="/home/dell/Desktop/prepared_data"
    img_size=48
    batch=448
    batch_size=[192,64,64,128]
    
    addr=[os.path.join(base_dir,"DATA/%d/neg_%d_train.tfrecords"%(img_size,img_size)),
          os.path.join(base_dir,"DATA/%d/pos_%d_train.tfrecords"%(img_size,img_size)),
          os.path.join(base_dir,"DATA/%d/par_%d_train.tfrecords"%(img_size,img_size)),
          os.path.join(base_dir,"DATA/%d/land_%d_train.tfrecords"%(img_size,img_size))]  

    model=Onet_model
    model_name="Onet_model"    
    train_step=100001
    learning_rate=0.001
    
    save_model_path=os.path.join(base_dir,"model/%s"%(model_name))
    
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path) 
        
    if(model_name=="Onet_model"):
        ratio=[1,0.5,1]
    else:
        ratio=[1,0.5,0.5]
    

    begin=time.time()        
    main(model)
    
# tensorboard --logdir=/home/dell/Desktop/prepared_data/model/Onet_model/