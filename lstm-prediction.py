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

max_vector_length = 30


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
trainX = scaler.fit_transform(trainX)
testX = scaler.fit_transform(testX)

#assert(test_dataset_array.ndim == 2)


#plt.plot(dataset)
#plt.show()
#trainX = numpy.array(train_dataset_array[:,0])
#testX = test_dataset_array[:,0]

# For predicting
look_back = 1
trainX, trainY = convert_dataset(trainX, look_back)
testX, testY = convert_dataset(testX, look_back)

# reshape input to be [samples, time steps, features]
#trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
#testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# Create and fit the LSTM network
embedding_vecor_length = 64
samples = 264080
hidden_layers = 4
max_features = 64122
embedding_size = 64

in_out_neurons = 30
hidden_neurons = 300
vector_len = 30

model = Sequential()
model.add(LSTM(3, input_shape=(look_back, max_vector_length)))
model.add(Dense(max_vector_length))
model.compile(loss="mean_squared_error", optimizer="adam", metrics=['accuracy'])
model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=2)

model.summary()

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY)

# calculate root mean squared error
#rmse = numpy.sqrt(((testPredict - trainY) ** 2).mean(axis=0))
print trainY
print '-----------'
print trainPredict[0]
print '-----------'
print trainPredict[:,0]

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()