# -*- coding: utf-8 -*-
"""
@author: friedhelm

"""
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
        train_accuracy=cal_accuracy(_label,label)
        tf.summary.scalar('accuracy',train_accuracy) 

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
        
        writer_train=tf.summary.FileWriter('C:\\Users\\312\\Desktop\\',sess.graph)
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
                    print('loss',sess.run(loss,{image:image_batch,label:label_batch,roi:roi_batch,landmark:landmark_batch}))               
                    print('time',time.time()-begin)
                    if(i%10000==0):
                        saver.save(sess,"E:\\friedhelm\\object\\face_detection_MTCNN\\%s\\%s.ckpt"%(model_name,model_name),global_step=i)
        except  tf.errors.OutOfRangeError:
            print("finished")
        finally:
            coord.request_stop()
            writer_train.close()
        coord.join(threads)
    
def main(model):
    
    with tf.name_scope('input'):
        image=tf.placeholder(tf.float32,name='image')
        label=tf.placeholder(tf.int32,name='label')
        roi=tf.placeholder(tf.float32,name='roi')
        landmark = tf.placeholder(tf.float32,name='landmark')  

    train(image,label,roi,landmark,model,model_name)

if __name__=='__main__':
    
    img_size=24
    batch=448
    batch_size=[192,64,64,128]
    addr=["E:\\friedhelm\\object\\face_detection_MTCNN\\DATA\\%d\\neg_%d_train.tfrecords"%(img_size,img_size),
          "E:\\friedhelm\\object\\face_detection_MTCNN\\DATA\\%d\\pos_%d_train.tfrecords"%(img_size,img_size),
          "E:\\friedhelm\\object\\face_detection_MTCNN\\DATA\\%d\\par_%d_train.tfrecords"%(img_size,img_size),
          "E:\\friedhelm\\object\\face_detection_MTCNN\\DATA\\%d\\land_%d_train.tfrecords"%(img_size,img_size)]  

    model=Rnet_model
    model_name="Rnet_model"    
    train_step=100001
    learning_rate=0.001
    
    save_model_path="E:\\friedhelm\\object\\face_detection_MTCNN\\%s"%(model_name)
    
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path) 
        
    if(model_name=="Onet_model"):
        ratio=[1,0.5,1]
    else:
        ratio=[1,0.5,0.5]
    

    begin=time.time()        
    main(model)
    
# tensorboard --logdir=C:\\Users\\312\\Desktop\\