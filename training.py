#!/usr/bin/env python2
# -*- coding: utf-8 -*-


# This is a re-implementation of training code of our paper:
# X. Fu, J. Huang, D. Zeng, Y. Huang, X. Ding and J. Paisley. “Removing Rain from Single Images via a Deep Detail Network”, CVPR, 2017.
# author: Xueyang Fu (fxy@stu.xmu.edu.cn)

import os
import re
import time
import numpy as np
import tensorflow as tf
from network import inference
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(description="ddn_train")
parser.add_argument("--batch_size", type=int, default=18, help="Training batch size")
parser.add_argument("--iterations", type=int, default=120000, help="Number of training iteration")
parser.add_argument("--save_path", type=str, default="./model/Rain100L/", help='path to save models and log files')
parser.add_argument("--data_path",type=str, default="./TrainData/Rain100L",help='path to training data')

opt = parser.parse_args()

##################### Select GPU device ####################################
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
############################################################################

tf.reset_default_graph()

##################### Network parameters ###################################
patch_size = 64              # patch size 
KernelSize = 3               # kernel size 
learning_rate = 0.1          # learning rate
iterations = opt.iterations   # iterations
batch_size = opt.batch_size    # batch size
save_model_path = opt.save_path # saved model's path
model_name = 'model-epoch'   # saved model's name
############################################################################


input_path = opt.data_path + "/input/"    # the path of rainy images
gt_path = opt.data_path + "/label/"       # the path of ground truth


input_files = os.listdir(input_path)
gt_files = os.listdir(gt_path) 
 
# randomly select image patches
def _parse_function(filename, label):  
     
  image_string = tf.read_file(filename)  
  image_decoded = tf.image.decode_jpeg(image_string, channels=3)  
  rainy = tf.cast(image_decoded, tf.float32)/255.0
  
  
  image_string = tf.read_file(label)  
  image_decoded = tf.image.decode_jpeg(image_string, channels=3)  
  label = tf.cast(image_decoded, tf.float32)/255.0

  t = time.time()
  rainy = tf.random_crop(rainy, [patch_size, patch_size ,3],seed = t)   # randomly select patch
  label = tf.random_crop(label, [patch_size, patch_size ,3],seed = t)   
  return rainy, label 






if __name__ == '__main__':   
   RainName = os.listdir(input_path)
   for i in range(len(RainName)):
      RainName[i] = input_path + RainName[i]
      
   LabelName = os.listdir(gt_path)    
   for i in range(len(LabelName)):
       LabelName[i] = gt_path + LabelName[i] 
    
   filename_tensor = tf.convert_to_tensor(RainName, dtype=tf.string)  
   labels_tensor = tf.convert_to_tensor(LabelName, dtype=tf.string)   
   dataset = tf.data.Dataset.from_tensor_slices((filename_tensor, labels_tensor))
   dataset = dataset.map(_parse_function)    
   dataset = dataset.prefetch(buffer_size=batch_size * 10)
   dataset = dataset.batch(batch_size).repeat()  
   iterator = dataset.make_one_shot_iterator()
   
   rainy, labels = iterator.get_next()     
   
   
   outputs = inference(rainy, is_training = True)
   loss = tf.reduce_mean(tf.square(labels - outputs))    # MSE loss

   
   lr_ = learning_rate
   lr = tf.placeholder(tf.float32 ,shape = [])  

   update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
   with tf.control_dependencies(update_ops):
        train_op =  tf.train.MomentumOptimizer(lr, 0.9).minimize(loss) 

   
   all_vars = tf.trainable_variables()   
   g_list = tf.global_variables()
   bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
   bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
   all_vars += bn_moving_vars
   print("Total parameters' number: %d" %(np.sum([np.prod(v.get_shape().as_list()) for v in all_vars])))  
   saver = tf.train.Saver(var_list=all_vars, max_to_keep=5)
   
   
   config = tf.ConfigProto()
   config.gpu_options.per_process_gpu_memory_fraction = 0.8 # GPU setting
   config.gpu_options.allow_growth = True
   init =  tf.group(tf.global_variables_initializer(), 
                         tf.local_variables_initializer())  
    
   with tf.Session(config=config) as sess:      
      with tf.device('/gpu:0'): 
            sess.run(init)
            tf.get_default_graph().finalize()
            print("Load model from:", save_model_path)
            if tf.train.get_checkpoint_state(save_model_path):   # load previous trained models
               ckpt = tf.train.latest_checkpoint(save_model_path)
               saver.restore(sess, ckpt)
               ckpt_num = re.findall(r'(\w*[0-9]+)\w*',ckpt)
               start_point = int(ckpt_num[0]) + 1   
               print("successfully load previous model")
       
            else:   # re-training if no previous trained models
               start_point = 0    
               print("re-training")
    
    
            check_data, check_label = sess.run([rainy, labels])
            print("Check patch pair:")  
            plt.subplot(1,2,1)     
            plt.imshow(check_data[0,:,:,:])
            plt.title('input')         
            plt.subplot(1,2,2)    
            plt.imshow(check_label[0,:,:,:])
            plt.title('ground truth')        
            plt.show()
    
    
            start = time.time()  
            
            for j in range(start_point,iterations):   #  iterations
                if j+1 > int(1e5):
                    lr_ = learning_rate*0.1
                if j+1 > int(2e5):
                    lr_ = learning_rate*0.01             
                    
    
                _,Training_Loss = sess.run([train_op,loss], feed_dict={lr: lr_}) # training
          
                if np.mod(j+1,100) == 0 and j != 0: # save the model every 100 iterations
                   end = time.time()              
                   print ('%d / %d iteraions, learning rate = %.3f, Training Loss = %.4f, runtime = %.1f s' 
                          % (j+1, iterations, lr_, Training_Loss, (end - start)))                  
                   save_path_full = os.path.join(save_model_path, model_name) # save model
                   saver.save(sess, save_path_full, global_step = j+1, write_meta_graph=False)
                   start = time.time()
                   
            print('Training is finished.')
   sess.close()  
