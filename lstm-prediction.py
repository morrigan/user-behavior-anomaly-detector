import helpers
import numpy
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import plot_model
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

# Create datasets
train_dataset = pandas.read_csv("./data/merged.csv", delimiter=';', engine='python')
test_dataset = pandas.read_csv("./data/test.csv", delimiter=';', engine='python')

train_dataset_array = collection_values_to_array(train_dataset)
test_dataset_array = collection_values_to_array(test_dataset)

assert(test_dataset_array.ndim == 2)


#plt.plot(dataset)
#plt.show()
#trainX = numpy.array(train_dataset_array[:,0])
#testX = test_dataset_array[:,0]

# For predicting
look_back = 3
trainX, trainY = convert_dataset(train_dataset_array, look_back)
testX, testY = convert_dataset(test_dataset_array, look_back)

# reshape input to be [samples, time steps, features]
#trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
#testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

print trainX.shape

# Create and fit the LSTM network
embedding_vecor_length = 64
samples = 179580
max_sentence_length = 30
hidden_layers = 4


model = Sequential()
model.add(LSTM(4, input_shape=(look_back, max_sentence_length)))
model.add(Dense(max_sentence_length))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=32, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(trainX)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(trainX)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(trainX)-1, :] = testPredict
# plot baseline and predictions
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()