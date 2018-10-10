from model_mnist import CGAN
import tensorflow as tf
from utils import Mnist
import os

flags = tf.app.flags

flags.DEFINE_string("sample_dir" , "samples_for_test" , "the dir of sample images")
flags.DEFINE_integer("output_size", 28 , "the size of generate image")
flags.DEFINE_float("learn_rate", 0.0002, "the learning rate for gan")
flags.DEFINE_integer("batch_size", 64, "the batch number")
flags.DEFINE_integer("z_dim", 100, "the dimension of noise z")
flags.DEFINE_integer("y_dim", 10, "the dimension of condition y")
flags.DEFINE_string("log_dir" , "/tmp/tensorflow_mnist" , "the path of tensorflow's log")
flags.DEFINE_string("model_path" , "model/model.ckpt" , "the path of model")
flags.DEFINE_string("visua_path" , "visualization" , "the path of visuzation images")
flags.DEFINE_integer("op" , 0, "0: train ; 1:test ; 2:visualize")

FLAGS = flags.FLAGS
#
if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)
if not os.path.exists(FLAGS.log_dir):
    os.makedirs(FLAGS.log_dir)
if not os.path.exists(FLAGS.model_path):
    os.makedirs(FLAGS.model_path)
if not os.path.exists(FLAGS.visua_path):
    os.makedirs(FLAGS.visua_path)

def main(_):

    mn_object = Mnist()

    cg = CGAN(data_ob = mn_object, sample_dir = FLAGS.sample_dir, output_size=FLAGS.output_size, learn_rate=FLAGS.learn_rate
         , batch_size=FLAGS.batch_size, z_dim=FLAGS.z_dim, y_dim=FLAGS.y_dim, log_dir=FLAGS.log_dir
         , model_path=FLAGS.model_path, visua_path=FLAGS.visua_path)

    cg.build_model()

    if FLAGS.op == 0:

        cg.train()

    elif FLAGS.op == 1:

        cg.test()

    else:

        cg.visual()

if __name__ == '__main__':
    tf.app.run()
