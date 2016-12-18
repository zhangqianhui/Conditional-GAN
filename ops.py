import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm

#the implements of leakyRelu
def lrelu(x , alpha = 0.2 , name="LeakyReLU"):
    return tf.maximum(x , alpha*x)

def conv2d(x, W , b , strides=2):

    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x , W , strides=[1, strides , strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)

    return x

def de_conv(x , W , b , out_shape):
    with tf.name_scope('deconv') as scope:
        deconv = tf.nn.conv2d_transpose(x , W ,
        out_shape , [1 , 2 , 2 , 1] , padding='SAME', name=None)
        out = tf.nn.bias_add(deconv , b)
        return out

def fully_connect(x , weight , bias):
    return tf.add(tf.matmul(x , weight) , bias)

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()

    print x_shapes
    print y_shapes

    return tf.concat(3 , [x , y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2] , y_shapes[3]])])

def batch_normal(input , scope="scope" , reuse=False):
    return batch_norm(input , epsilon=1e-5, decay=0.9 , scale=True, scope=scope , reuse = reuse , updates_collections=None)








