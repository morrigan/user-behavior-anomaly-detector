import helpers
import numpy
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import plot_model
import pandas

# Convert an array of values into a dataset matrix
def convert_dataset(dataset, look_back = 3):
	dataX, dataY = [], []
	for i in range(len(dataset) - look_back - 1):
		a = dataset[i:(i + look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 1])
	return numpy.array(dataX), numpy.array(dataY)


def collection_values_to_array(dataset):
	dataset = numpy.array(dataset)
	new_dataset = []
	for row in dataset:
		row_array = numpy.array([numpy.array(eval(row[0]), dtype=int), row[1]])
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
trainX = numpy.array(train_dataset_array[:,0])
trainY = train_dataset_array[:,1]
testX = test_dataset_array[:,0]
testY = test_dataset_array[:,1]

# For predicting
#look_back = 3
#trainX, trainY = convert_dataset(train_dataset_array, look_back)
#testX, testY = convert_dataset(test_dataset_array, look_back)


# reshape input to be [samples, time steps, features]
#trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
#testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# Create and fit the LSTM network
embedding_vecor_length = 64
samples = 179580
max_sentence_length = 50
hidden_layers = 4

print trainX[0]

trainX = sequence.pad_sequences(trainX, maxlen=max_sentence_length)
testX = sequence.pad_sequences(testX, maxlen=max_sentence_length)

print trainX[0]
print trainX.shape

model = Sequential()
model.add(Embedding(samples, embedding_vecor_length, input_length=max_sentence_length))
model.add(LSTM(hidden_layers))
#model.add(Dropout(.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(trainX, trainY, validation_data=(testX, testY), epochs=1, batch_size=64)

scores = model.evaluate(testX, testY, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

plot_model(model, to_file='model.png')


