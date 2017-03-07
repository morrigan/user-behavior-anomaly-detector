import tensorflow as tf
import numpy as np
import helpers
import LSTM
import pprint
import os
pp = pprint.PrettyPrinter()

flags = tf.app.flags
flags.DEFINE_string(
  "input_file", os.path.abspath("./data/train.tfrecords"),
  "Input file containing Examples"
)
flags.DEFINE_float("learning_rate", 0.005, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("keep_prob", 0.9, "Keep Probability for dropout")
flags.DEFINE_integer("batch_size", 128, "The size of batch [128]")
flags.DEFINE_integer("n_recurrent_layers", 2, "Number of recurrent layers [3]")
flags.DEFINE_integer("n_fc_layers", 2, "Number of fully connected layers [2]")
flags.DEFINE_integer("recurrent_layer_width", 128, "Width of recurrent layers [256]")
flags.DEFINE_integer("fc_layer_width", 128, "Width of fully connected layers [256]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing.")
flags.DEFINE_integer("epoch", 50, "Epoch to train. [50]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints.")
FLAGS = flags.FLAGS

def load_batches(filename):
    # get single examples
    filename_queue = tf.train.string_input_producer([filename], num_epochs=None)
    action, label = helpers.read_single_example(filename_queue)
    # groups examples into batches randomly
    actions_batch, labels_batch = tf.train.shuffle_batch(
        [action, label], batch_size=FLAGS.batch_size,
        capacity=2000,
        min_after_dequeue=1000)

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)
    actions, labels = sess.run([actions_batch, labels_batch])

def main(_):
    #pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    load_batches(FLAGS.input_file)

if __name__ == '__main__':
    tf.app.run()