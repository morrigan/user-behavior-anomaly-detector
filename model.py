from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
from helpers import *


class LSTM(object):
    def __init__(self, sess, batch_size, vocab_size,
                 keep_prob, max_length, n_recurrent_layers, n_fc_layers,
                 recurrent_layer_width, fc_layer_width, checkpoint_dir, epoch):
        """Init variables, builds model"""
        self.sess = sess
        self.vocab_size = vocab_size
        self.n_classes = 1 #n_classes
        self.batch_size = batch_size
        self.max_length = max_length
        self.n_recurrent_layers = n_recurrent_layers
        self.n_fc_layers = n_fc_layers
        self.recurrent_layer_width = recurrent_layer_width
        self.fc_layer_width = fc_layer_width
        self.checkpoint_dir = checkpoint_dir
        self.epoch = epoch

        self.build_model()

    def build_model(self):
        """Builds model and creates loss functions"""

        # X is the input data, a tensor of actions
        # where each action is a list of 1-hot encodings of words
        # [[[0, 0, 1, 0...], [1, 0, 1, 0...]...],[...]]
        self.X = tf.placeholder(tf.float32, [None, self.max_length, self.vocab_size])   # check if vocab_size ok

        # y are labels
        # [[1, 0, 0], [0, 0, 1]]
        self.y = tf.placeholder(tf.float32, [None, self.n_classes], name='y')

        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # logits is our predictions
        # will be like [[1, 0, 0], [0, 0, 1]]
        self.logits = self.rnn_simple_model(self.X, self.y)

        # Construct loss function (mean of all cross-entropy losses)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.logits, self.y))
        # self.loss = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.logits), reduction_indices=[1]))
        self.loss_sum = tf.scalar_summary("loss", self.loss)

        # Print trainable variables (useful for debugging)
        print("Trainable Variables:", [var.name for var in tf.trainable_variables()])
        self.t_vars = tf.trainable_variables()

        self.saver = tf.train.Saver()

    def train(self, config, X_train, y_train, X_test, y_test):
        """Train DeepPDF"""

        train_step = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.loss)
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.sess.run(tf.initialize_all_variables())

        self.writer = tf.train.SummaryWriter("./logs", self.sess.graph)
        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in range(self.epoch):
            all_minibatch_indices = get_random_minibatch_indices(len(X_train), config.batch_size)
            num_minibatches = len(all_minibatch_indices)

            for minibatch_idx in range(num_minibatches):
                minibatch_indices = all_minibatch_indices[minibatch_idx]
                batch_x, batch_y = np.zeros([config.batch_size] + [X_train.shape][:-1], dtype='float32'), np.zeros(
                    [config.batch_size] + [y_train.shape][:-1], dtype='float32')
                batch_x = X_train[minibatch_indices, :, :]
                batch_y = y_train[minibatch_indices, :]

                # Update model weights
                # train_step.run(feed_dict={self.X:batch_x, self.y:batch_y, self.keep_prob: config.keep_prob})
                optimizer = tf.train.GradientDescentOptimizer(0.0002).minimize(self.loss)
                # self.sess.run([train_step], feed_dict={self.X:batch_x, self.y:batch_y, self.keep_prob: config.keep_prob})
                self.sess.run([optimizer],
                              feed_dict={self.X: batch_x, self.y: batch_y, self.keep_prob: config.keep_prob})

                batch_loss = self.sess.run(self.loss, feed_dict={self.X: batch_x, self.y: batch_y, self.keep_prob: 1.0})

                tf.scalar_summary('batch_loss_cross_entropy', self.loss)

                counter += 1
                if np.mod(counter, 5) == 1:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, batch loss: %.8f" \
                          % (epoch, minibatch_idx, num_minibatches,
                             time.time() - start_time, batch_loss))

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)

            with tf.name_scope('accuracy-metrics'):
                # Eval Accuracy after each epoch
                test_accuracy = accuracy.eval(feed_dict={self.X: X_test, self.y: y_test, self.keep_prob: 1.0})
                print("Epoch %d, testing accuracy %g" % (epoch, test_accuracy))

                test_loss = self.loss.eval(feed_dict={self.X: X_test, self.y: y_test, self.keep_prob: 1.0})
                print("Epoch %d, test loss %g" % (epoch, test_loss))

                training_accuracy = accuracy.eval(feed_dict={self.X: X_train, self.y: y_train, self.keep_prob: 1.0})
                print("Epoch %d, training accuracy %g" % (epoch, training_accuracy))

                training_loss = self.loss.eval(feed_dict={self.X: X_train, self.y: y_train, self.keep_prob: 1.0})
                print("Epoch %d, training loss %g" % (epoch, training_loss))

                tf.scalar_summary('test_accuracy', test_accuracy)
                tf.scalar_summary('test_loss', batch_loss)
                tf.scalar_summary('training_accuracy', batch_loss)
                tf.scalar_summary('training_loss', batch_loss)

                tf.scalar_summary('accuracy-metrics', accuracy)
        merged = tf.merge_all_summaries()

    def rnn_simple_model(self, X, y):
        """Returns Model Graph"""
        #print(X.get_shape())
        with tf.name_scope("recurrent_layers") as scope:
            # Create LSTM Cell
            cell = tf.contrib.rnn.core_rnn_cell.LSTMCell(self.recurrent_layer_width)
            cell = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            stacked_cells = tf.contrib.rnn.core_rnn_cell.MultiRNNCell([cell] * self.n_recurrent_layers)
            output, encoding = tf.nn.dynamic_rnn(stacked_cells, X, dtype=tf.float32)

        with tf.name_scope("fc_layers") as scope:
            # Connect RNN Embedding output into fully connected layers
            prev_layer = encoding
            for fc_index in range(0, self.n_fc_layers - 1):
                fci = lrelu(linear(prev_layer, self.fc_layer_width, 'fc{}'.format(fc_index)))
                fc_prev = fci

            #print fc_prev
            fc_final = linear(fc_prev, self.n_classes, 'fc{}'.format(self.n_fc_layers - 1))

        return tf.nn.softmax(fc_final)

    def save(self, checkpoint_dir, step):
        """Saves Model"""
        model_name = "sentiment.model"
        model_dir = "sentiment-%s" % (self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        """Loads Model"""
        print(" [*] Reading checkpoints...")

        model_dir = "sentiment-%s" % (self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False