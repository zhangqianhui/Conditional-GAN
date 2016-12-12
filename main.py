from model_mnist import dcgan
import tensorflow as tf
import os

flags = tf.app.flags
flags.DEFINE_string("dataset" , "mnist" , "the dataset for read images")
flags.DEFINE_string("sample_dir" , "samples_for_test" , "the dir of sample images")
flags.DEFINE_integer("output_size" , 28 , "the size of generate image")
flags.DEFINE_string("log_dir" , "/tmp/tensorflow_mnist" , "the path of tensorflow's log")
flags.DEFINE_string("model_path" , "model/model.ckpt" , "the path of model")
flags.DEFINE_string("visua_path" , "visualization" , "the path of visuzation images")
flags.DEFINE_integer("operation" , 0 , "0 : trian ; 1:test ; 2:visualize")

FLAGS = flags.FLAGS
#
if os.path.exists(FLAGS.sample_dir) == False:
    os.makedirs(FLAGS.sample_dir)
if os.path.exists(FLAGS.log_dir) == False:
    os.makedirs(FLAGS.log_dir)
if os.path.exists(FLAGS.model_path) == False:
    os.makedirs(FLAGS.model_path)
if os.path.exists(FLAGS.visua_path) == False:
    os.makedirs(FLAGS.visua_path)

def main(_):
    dcgan(operation = FLAGS.operation ,data_name=FLAGS.dataset ,  output_size=FLAGS.output_size , sample_path=FLAGS.sample_dir , log_dir=FLAGS.log_dir
           , model_path= FLAGS.model_path , visua_path=FLAGS.visua_path)

if __name__ == '__main__':
    tf.app.run()
