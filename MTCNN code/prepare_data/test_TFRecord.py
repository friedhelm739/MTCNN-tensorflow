# -*- coding: utf-8 -*-
"""
@author: friedhelm

"""
import tensorflow as tf

filename_queue = tf.train.string_input_producer(["E:\\friedhelm\\object\\face_detection_MTCNN\\DATA\\48\\neg_48_train.tfrecords"],shuffle=True,num_epochs=1)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue) #返回文件名和文件

features = tf.parse_single_example(serialized_example,
                               features={
                               'img':tf.FixedLenFeature([],tf.string),
                               'label':tf.FixedLenFeature([],tf.int64),                                   
                               'roi':tf.FixedLenFeature([4],tf.float32),
                               'landmark':tf.FixedLenFeature([10],tf.float32),
                               })
img=tf.decode_raw(features['img'],tf.uint8)
label=tf.cast(features['label'],tf.int32)
roi=tf.cast(features['roi'],tf.float32)
landmark=tf.cast(features['landmark'],tf.float32)
img = tf.reshape(img, [48,48,3])   
#     img=img_preprocess(img)
min_after_dequeue = 10000
batch_size = 64
capacity = min_after_dequeue + 10 * batch_size
image_batch, label_batch, roi_batch, landmark_batch = tf.train.shuffle_batch([img,label,roi,landmark], 
                                                    batch_size=batch_size, 
                                                    capacity=capacity, 
                                                    min_after_dequeue=min_after_dequeue,
                                                    num_threads=7)  


i=0
with tf.Session() as sess:
    sess.run((tf.global_variables_initializer(),
              tf.local_variables_initializer()))
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)    
    while(1):
        i=i+1
        if(i%9==1):
            print(sess.run(label_batch))