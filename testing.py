#!/usr/bin/env python2
# -*- coding: utf-8 -*-


# This is a re-implementation of testing code of this paper:
# X. Fu, J. Huang, D. Zeng, Y. Huang, X. Ding and J. Paisley. “Removing Rain from Single Images via a Deep Detail Network”, CVPR, 2017.
# author: Xueyang Fu (fxy@stu.xmu.edu.cn)


import argparse

import os
import time
import skimage.io
import numpy as np
import tensorflow as tf
from network import inference
import matplotlib.pyplot as plt

import time 


parser = argparse.ArgumentParser(description="ddn_test")
parser.add_argument("--input_path", type=str, default="./TestData/input_100L/", help='path to save models and log files')
parser.add_argument("--result_path",type=str, default="./TestData/results_100L/", help='path to training data')
parser.add_argument("--model_path",type=str, default="./model/model_L", help='path to training data')

opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # select GPU device


tf.reset_default_graph()

model_path = opt.model_path
pre_trained_model_path = './model/trained/model'


img_path = opt.input_path # the path of testing images
results_path = opt.result_path # the path of de-rained images
avg_time = 0
counter = 0

def _parse_function(filename):   
  image_string = tf.read_file(filename)  
  image_decoded = tf.image.decode_jpeg(image_string, channels=3)  
  rainy = tf.cast(image_decoded, tf.float32)/255.0 
  return rainy 


if __name__ == '__main__':
   imgName = os.listdir(img_path)
   num_img = len(imgName)
   
   whole_path = []
   for i in range(num_img):
      whole_path.append(img_path + imgName[i])
      
    
   filename_tensor = tf.convert_to_tensor(whole_path, dtype=tf.string)     
   dataset = tf.data.Dataset.from_tensor_slices((filename_tensor))
   dataset = dataset.map(_parse_function)    
   dataset = dataset.prefetch(buffer_size=10)
   dataset = dataset.batch(batch_size=1).repeat()  
   iterator = dataset.make_one_shot_iterator()
   
   rain = iterator.get_next() 


   output = inference(rain, is_training = False)
   output = tf.clip_by_value(output, 0., 1.)
   output = output[0,:,:,:]

   config = tf.ConfigProto()
   config.gpu_options.allow_growth=True   
   saver = tf.train.Saver()


   with tf.Session(config=config) as sess: 
      with tf.device('/gpu:0'): 
          if tf.train.get_checkpoint_state(model_path):  
              ckpt = tf.train.latest_checkpoint(model_path)  # try your own model 
              saver.restore(sess, ckpt)
              print ("Loading model")
          else:
             saver.restore(sess, pre_trained_model_path) # try a pre-trained model 
             print ("Loading pre-trained model")
	  
          count = 0 
          time_test = 0
          for i in range(num_img):   
             start_time = time.time()  
             derained, ori = sess.run([output, rain])              
             derained = np.uint8(derained* 255.)
             end_time = time.time()
             dur_time = end_time - start_time
             time_test += dur_time
             count += 1
             index = imgName[i].rfind('.')
             name = imgName[i][:index]
             skimage.io.imsave(results_path + name +'.png', derained)         
             print('%d / %d images processed' % (i+1,num_img))
      print('Avg. time:', time_test/count)
      print('All done')\

   sess.close()   
   
   plt.subplot(1,2,1)     
   plt.imshow(ori[0,:,:,:])          
   plt.title('rainy')
   plt.subplot(1,2,2)    
   plt.imshow(derained)
   plt.title('derained')
   plt.show()     
