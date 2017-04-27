from keras.callbacks import ModelCheckpoint

import helpers
import numpy
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Reshape
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
import pandas

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


# Parameters
max_vector_length = 30
hidden_layers = 4	# memory units
dropout = 0.2	# 20% probability
weights_file = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
###################

# Create datasets
train_dataset = pandas.read_csv("./data/test.csv", delimiter=';', engine='python')
test_dataset = pandas.read_csv("./data/primjer.csv", delimiter=';', engine='python')

# Convert strings
train_dataset_array = collection_values_to_array(train_dataset)
test_dataset_array = collection_values_to_array(test_dataset)

# Padding (from left)
trainX = sequence.pad_sequences(train_dataset_array, maxlen=max_vector_length, padding='pre')
testX = sequence.pad_sequences(test_dataset_array, maxlen=max_vector_length, padding='pre')

# normalize the dataset
trainX.reshape(-1, max_vector_length)
testX.reshape(-1, max_vector_length)

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = numpy.concatenate((trainX, testX), axis=0)
trainX_length = len(trainX)
dataset = scaler.fit_transform(dataset)
trainX, testX = dataset[0:trainX_length,:], dataset[trainX_length:len(dataset),:]

#assert(test_dataset_array.ndim == 2)


#plt.plot(dataset)
#plt.show()
#trainX = numpy.array(train_dataset_array[:,0])
#testX = test_dataset_array[:,0]

# For predicting
look_back = 1	# Doesn't work yet with others
trainX, trainY = convert_dataset(trainX, look_back)
testX, testY = convert_dataset(testX, look_back)

# reshape input to be [samples, time steps, features]
#trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
#testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# Create and fit the LSTM network
model = Sequential()
model.add(LSTM(hidden_layers, input_shape=(look_back, max_vector_length)))
model.add(Dense(max_vector_length))
#model.load_weights(weights_file)
model.compile(loss="mean_squared_error", optimizer="adam", metrics=['accuracy'])

model.summary()

# define the checkpoint for saving weights
checkpoint = ModelCheckpoint(weights_file, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# fit the model
model.fit(trainX, trainY, epochs=1, batch_size=100, verbose=2)


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

#plot_model(model, to_file='model.png')	# Visualize layers


print testPredict
print '----------'

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY)

print testPredict
print '----------'
print testY