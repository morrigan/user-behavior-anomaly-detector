import helpers
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Dense, Activation, Reshape
from keras.layers.embeddings import Embedding
from keras.layers import advanced_activations
from keras.preprocessing import sequence
from keras import callbacks
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
import pandas

def visualize_model_training(history):
	# list all data in history
	print history.history

	# summarize history for accuracy
	plt.plot(history.history['acc'])
	#plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

	# summarize history for loss
	plt.plot(history.history['loss'])
	#plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

def save_model(model):
	# serialize model to JSON
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)

	# serialize weights to HDF5
	model.save_weights("model.h5")
	print("Saved model to disk")

def create_model():
	model = Sequential()
	model.add(LSTM(hidden_layers, input_shape=(look_back, max_vector_length)))
	model.add(Dense(max_vector_length))
	model.add(advanced_activations.LeakyReLU(alpha=0))

	return model

def load_model():
	# load json and create model
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)

	# load weights into new model
	loaded_model.load_weights("model.h5")
	print("Loaded model from disk")
	return loaded_model

# Parameters
load_existing_model = False
max_vector_length = 30	# "features"
hidden_layers = 4	# memory units
dropout = 0.2
weights_file = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
###################

# Create datasets
train_dataset = pandas.read_csv("./data/test.csv", delimiter=';', engine='python')
test_dataset = pandas.read_csv("./data/primjer.csv", delimiter=';', engine='python')

# Convert strings
train_dataset_array = helpers.collection_values_to_array(train_dataset)
test_dataset_array = helpers.collection_values_to_array(test_dataset)

# Padding (from left, otherwise results are affected in Keras)
trainX = sequence.pad_sequences(train_dataset_array, maxlen=max_vector_length, padding='pre')
testX = sequence.pad_sequences(test_dataset_array, maxlen=max_vector_length, padding='pre')

# normalize the dataset
trainX.reshape(-1, max_vector_length)
testX.reshape(-1, max_vector_length)

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = numpy.concatenate((trainX, testX), axis=0)
trainX_length = len(trainX)
print dataset
print dataset.shape
dataset = scaler.fit_transform(dataset)
trainX, testX = dataset[0:trainX_length,:], dataset[trainX_length:len(dataset),:]

print "TestX after fitting:"
print testX
print "---------------------\n"
#assert(test_dataset_array.ndim == 2)


#plt.plot(dataset)
#plt.show()
#trainX = numpy.array(train_dataset_array[:,0])
#testX = test_dataset_array[:,0]

# For predicting
look_back = 1	# Doesn't work yet with others
trainX, trainY = helpers.convert_dataset(trainX, look_back)
testX, testY = helpers.convert_dataset(testX, look_back)

# reshape input to be [samples, time steps, features]
#trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
#testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# LSTM network
if (load_existing_model == True):
	model = load_model()
else:
	model = create_model()
model.compile(loss="mean_squared_error", optimizer="adam", metrics=['accuracy'])
model.summary()

# Tensorboard callback
#tbCallBack = callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

# Fit the model
history = model.fit(trainX, trainY, epochs=1, batch_size=24, verbose=2, callbacks=[])

# Save current model if existing is not loaded
if (load_existing_model == False):
	save_model(model)

# Make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Visualize
#plot_model(model, to_file='model.png')
#visualize_model_training(history)

print "Test predict before transforming back"
print testPredict
print '----------'
#testPredict = numpy.around(testPredict, decimals=1)
#print testPredict
#print testPredict.shape

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY)

print "Test predict after transform"
print testPredict
print testY

