import tensorflow as tf
from GuidedFilter import guided_filter


num_feature = 16             # number of feature maps
KernelSize = 3               # kernel size 
num_channels = 3             # number of input's channels 


# network structure
def inference(images, is_training):
    regularizer = tf.contrib.layers.l2_regularizer(scale = 1e-10)
    initializer = tf.contrib.layers.xavier_initializer()

    base = guided_filter(images, images, 15, 1, nhwc=True) # using guided filter for obtaining base layer
    detail = images - base   # detail layer

   #  layer 1
    with tf.variable_scope('layer_1'):
         output = tf.layers.conv2d(detail, num_feature, KernelSize, padding = 'same', kernel_initializer = initializer, 
                                   kernel_regularizer = regularizer, name='conv_1')
         output = tf.layers.batch_normalization(output, training=is_training, name='bn_1')
         output_shortcut = tf.nn.relu(output, name='relu_1')
  
   #  layers 2 to 25
    for i in range(12):
        with tf.variable_scope('layer_%d'%(i*2+2)):	
             output = tf.layers.conv2d(output_shortcut, num_feature, KernelSize, padding='same', kernel_initializer = initializer, 
                                       kernel_regularizer = regularizer, name=('conv_%d'%(i*2+2)))
             output = tf.layers.batch_normalization(output, training=is_training, name=('bn_%d'%(i*2+2)))	
             output = tf.nn.relu(output, name=('relu_%d'%(i*2+2)))


        with tf.variable_scope('layer_%d'%(i*2+3)): 
             output = tf.layers.conv2d(output, num_feature, KernelSize, padding='same', kernel_initializer = initializer,
                                       kernel_regularizer = regularizer, name=('conv_%d'%(i*2+3)))
             output = tf.layers.batch_normalization(output, training=is_training, name=('bn_%d'%(i*2+3)))
             output = tf.nn.relu(output, name=('relu_%d'%(i*2+3)))

        output_shortcut = tf.add(output_shortcut, output)   # shortcut
    
    # layer 26
    with tf.variable_scope('layer_26'):
         output = tf.layers.conv2d(output_shortcut, num_channels, KernelSize, padding='same',   kernel_initializer = initializer, 
                                   kernel_regularizer = regularizer, name='conv_26')
         neg_residual = tf.layers.batch_normalization(output, training=is_training, name='bn_26')

    final_out = tf.add(images, neg_residual)

    return final_out