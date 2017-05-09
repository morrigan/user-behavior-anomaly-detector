import helpers
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Dense, Activation, Reshape, Masking, Dropout
from keras.layers.embeddings import Embedding
from keras.layers import advanced_activations
from keras.preprocessing import sequence
from keras.utils import plot_model
import pandas, math

############ Parameters ###############
load_existing_model = False
save_training_model = True
visualize_model = False

max_vector_length = 30
hidden_layers = 128	# memory units
dropout = 0.2
batch_size = 32
epochs = 3
#######################################

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

def load_model(model_filename = 'model.json', weights_filename = "model.h5"):
    # load json and create model
    json_file = open(model_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(weights_filename)

    loaded_model.compile(loss="mean_squared_error", optimizer="adam", metrics=['accuracy'])
    loaded_model.summary()
    print "Loaded model from filesystem."
    return loaded_model

def create_model():
    model = Sequential()
    #model.add(Embedding(input_dim=64123, output_dim=3, input_length=max_vector_length, mask_zero=True))
    model.add(Masking(mask_value=0, input_shape=(1, max_vector_length)))
    model.add(LSTM(hidden_layers))
    model.add(Dense(max_vector_length))
    model.add(advanced_activations.LeakyReLU(alpha=0))

    return model

def calculate_score(actions, predictions, scores):
    totals = []
    for i in range(len(actions)-1):
        action = sequence.pad_sequences(numpy.array([actions[i]]), maxlen=max_vector_length, padding='pre')
        rmse = math.sqrt(mean_squared_error(action[0], predictions[i]))
        total = 1/rmse * 100 * scores[i]
        totals.append(total)

    return totals

def predict(model, actions):
    actions = numpy.array(actions)
    actions = sequence.pad_sequences(actions, maxlen=max_vector_length, padding='pre')

    assert (actions.shape[1] == max_vector_length)

    x, y = helpers.get_real_predictions(actions)

    x = numpy.reshape(x, (x.shape[0], 1, x.shape[1]))

    predicted = model.predict(x, batch_size=1, verbose=0)
    return predicted


def train_on_datasets():
    # Create datasets
    train_dataset = pandas.read_csv("./data/merged.csv", delimiter=';', engine='python')
    test_dataset = pandas.read_csv("./data/primjer.csv", delimiter=';', engine='python')

    # Convert strings
    train_dataset_array = helpers.collection_values_to_array(train_dataset)
    test_dataset_array = helpers.collection_values_to_array(test_dataset)	#shape (n,)

    # Padding (from left, otherwise results are affected in Keras)
    trainX = sequence.pad_sequences(train_dataset_array, maxlen=max_vector_length, padding='pre')	#shape (n, 30)
    testX = sequence.pad_sequences(test_dataset_array, maxlen=max_vector_length, padding='pre')

    assert (trainX.shape[1] == max_vector_length)

    # For predicting
    trainX, trainY = helpers.get_real_predictions(trainX)
    testX, testY = helpers.get_real_predictions(testX)

    print trainX

    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # LSTM network
    if (load_existing_model == True):
        model = load_model()
    else:
        model = create_model()
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=['accuracy'])
    model.summary()

    # Model training
    model.add(Dropout(dropout))
    history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=False)

    # Visualize model
    if (visualize_model == True):
        plot_model(model, to_file='model.png')
        visualize_model_training(history)

    # Save model
    if (save_training_model == True):
        save_model(model)

    # Make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    print "Test predict:"
    print testPredict
    print '----------'
    print "TestY"
    print testY
    #testPredict = numpy.around(testPredict, decimals=1)
    #print testPredict
    #print testPredict.shape

if __name__ == "__main__":
    train_on_datasets()
