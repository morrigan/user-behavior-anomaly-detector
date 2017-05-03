import np
import random
import tensorflow as tf
import numpy

def read_data(filename):
    actions = []
    lengths = []
    labels = []
    for serialized_example in tf.python_io.tf_record_iterator(filename):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)

        action = example.features.feature['action'].int64_list.value
        action_len = example.features.feature['action_len'].int64_list.value
        label = example.features.feature['label'].int64_list.value[0]
        actions.append(action)
        labels.append(label)
        lengths.append(action_len)
        #print action, action_len, label
    return actions, lengths, labels


def read_single_example(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'action': tf.FixedLenFeature([200], tf.int64),
            #'action_len': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([1], tf.int64),
        })
    print features['action']
    return features['action'], features['label']

def get_random_minibatch_indices(n_examples, batch_size):
    indices = range(n_examples)
    random.shuffle(indices)
    num_batches = n_examples/batch_size
    minibatch_indices = np.zeros(shape=(num_batches, batch_size), dtype='int64')
    for b_i in range(num_batches):
        for ex_i in range(batch_size):
            minibatch_indices[b_i] = indices[b_i*batch_size:(b_i+1)*batch_size]
    return minibatch_indices

## Operations
def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def linear(input_, output_size, layer_scope, stddev=0.02, bias_start=0.0):
    if (input_.count):
        print input_
        shape = input_.get_shape().as_list()
        print shape
    else:
        shape = input_


    with tf.variable_scope(layer_scope):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        return tf.matmul(input_, matrix) + bias


# Convert an array of values into a dataset matrix
def convert_dataset(dataset, look_back = 3):
	dataX, dataY = [], []
	for i in range(len(dataset) - look_back - 1):
		a = dataset[i:(i + look_back)]
		b = dataset[i + look_back]
		dataX.append(a)
		dataY.append(b)
	return numpy.array(dataX), numpy.array(dataY)


def collection_values_to_array(dataset):
	dataset = numpy.array(dataset)
	new_dataset = []
	for row in dataset:
		row_array = numpy.array(eval(row[0]))
		new_dataset.append(row_array)

	return numpy.array(new_dataset)


