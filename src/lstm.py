#!/usr/bin/python

import math
import matplotlib.pyplot as plt
import numpy
import pandas
from keras.layers import LSTM as LSTM_CELL, Dense, Masking, Dropout
from keras.layers import advanced_activations
from keras.models import Sequential, model_from_json
from keras.preprocessing import sequence
from keras.utils import plot_model
from sklearn.metrics import mean_squared_error

import helpers

class LSTM:
    def __init__(self, config_file):
        self.settings = helpers.getConfig(config_file)


    def visualize_model_training(self, history):
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

    def save_model(self, model):
        # serialize model to JSON
        model_json = model.to_json()
        with open(self.settings.get("LSTM", "model_filename"), "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        model.save_weights(self.settings.get("LSTM", "weights_filename"))
        print("Saved model to disk")

    def load_model(self):
        # load json and create model
        json_file = open(self.settings.get('LSTM', 'model_filename'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights(self.settings.get("LSTM", "weights_filename"))

        loaded_model.compile(loss="mean_squared_error", optimizer="adam", metrics=['accuracy'])
        loaded_model.summary()
        print "Loaded model from filesystem."
        return loaded_model

    def create_model(self):
        model = Sequential()
        #model.add(Embedding(input_dim=64123, output_dim=3, input_length=max_vector_length, mask_zero=True))
        model.add(Masking(mask_value=0, input_shape=(1, self.settings.getint("LSTM", "max_vector_length"))))
        model.add(LSTM_CELL(self.settings.getint("LSTM", "hidden_layers")))
        model.add(Dense(self.settings.getint('LSTM', 'max_vector_length')))
        model.add(advanced_activations.LeakyReLU(alpha=0))   # clamp all values below 0 to 0
        #model.add(Activation('relu'))

        return model

    def train_model(self, model, trainX, trainY, verbose=2):
        model.add(Dropout(self.settings.getfloat("LSTM", "dropout")))
        return model.fit(trainX, trainY, epochs=self.settings.getint("LSTM", "epochs"),
                         batch_size=self.settings.getint("LSTM", "batch_size"), verbose=verbose, shuffle=False)

    def calculate_score(self, actions, predictions, scores):
        totals = []
        for i in range(len(actions)-1):
            action = sequence.pad_sequences(numpy.array([actions[i]]), maxlen=self.settings.getint("LSTM", "max_vector_length"), padding='pre')
            rmse = math.sqrt(mean_squared_error(action[0], predictions[i]))
            # total = helpers.sigmoid(1/rmse * scores[i]) * 100
            total = round(rmse * scores[i] / 100, 2)
            totals.append(total)

        return totals

    def update_model(self, model, x, y):
        return self.train_model(model, x, y, verbose=0)

    def forecast(self, model, actions):
        if (len(actions) <= 1):
            print "At least 2 actions are needed to calculate and compare predictions."
            return 0,0

        max_vector_length = self.settings.getint("LSTM", "max_vector_length")
        actions = numpy.array(actions)
        actions = sequence.pad_sequences(actions, maxlen=max_vector_length, padding='pre')

        assert (actions.shape[1] == max_vector_length)

        x, y = helpers.get_real_predictions(actions)

        x = numpy.reshape(x, (x.shape[0], 1, x.shape[1]))

        predicted = model.predict(x, batch_size=1, verbose=0)
        history = self.update_model(model, x, y)   # save data
        return predicted, history

    def load_datasets(self):
        # Create datasets
        train_dataset = pandas.read_csv(self.settings.get('Data', 'train_dataset_file'), delimiter=';', engine='python')
        test_dataset = pandas.read_csv(self.settings.get('Data', 'test_dataset_file'), delimiter=';', engine='python')

        # Convert strings
        train_dataset_array = helpers.collection_values_to_array(train_dataset)
        test_dataset_array = helpers.collection_values_to_array(test_dataset)  # shape (n,)

        return train_dataset_array, test_dataset_array

    """
        Train dataset on a given model (existing or created) and make predictions.
        Return root mean squared error.
    """
    def train_on_dataset(self, train_dataset_array, model):
        max_vector_length = self.settings.getint("LSTM", "max_vector_length")

        train = helpers.padding(train_dataset_array, max_vector_length)  # shape (n, 30)

        assert (train.shape[1] == max_vector_length)

        if (self.settings.getboolean("LSTM", "scale_data") == True):
            scaler, trainX = helpers.scale(train)

        # For predicting
        trainX, trainY = helpers.get_real_predictions(trainX)
        # reshape input to be [samples, time steps, features]
        trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

        # Model training
        history = self.train_model(model, trainX, trainY)

        # Visualize model
        if (self.settings.getboolean("LSTM", "visualize_model") == True):
            plot_model(model, to_file='model.png')
            self.visualize_model_training(history)

        # Save model
        if (self.settings.getboolean("LSTM", "save_training_model") == True):
            self.save_model(model)

        # Make predictions
        trainPredict = model.predict(trainX)
        # Calculate error
        rmse = helpers.calculate_rmse(trainX, trainPredict)

        return rmse

    """
        Use loaded model to run a test dataset on it.
        Return root mean squared error.
    """
    def test_on_dataset(self, test_dataset_array, model):
        max_vector_length = self.settings.getint("LSTM", "max_vector_length")

        test = helpers.padding(test_dataset_array, max_vector_length)  # shape (n, 30)

        assert (test.shape[1] == max_vector_length)

        if (self.settings.getboolean("LSTM", "scale_data") == True):
            scaler, testX = helpers.scale(test)

        # For predicting
        testX, testY = helpers.get_real_predictions(testX)
        # reshape input to be [samples, time steps, features]
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        # Make predictions
        testPredict = model.predict(testX)
        # Calculate error
        rmse = helpers.calculate_rmse(testX, testPredict)

        return rmse

    """
        Use preprocessed dataset file for training and testing.
        Use model related configuration from settings.ini file.
    """
    def train_on_datasets(self, train_dataset_array = [], test_dataset_array = []):
        # If none given, load datasets from .csv defined in settings
        if (len(train_dataset_array) == 0 and len(test_dataset_array) == 0):
            train_dataset_array, test_dataset_array = self.load_datasets()

        # LSTM model
        if (self.settings.getboolean("LSTM", "load_existing_model") == True):
            model = self.load_model()
        else:
            model = self.create_model()
        model.compile(loss="mean_squared_error", optimizer="adam", metrics=['accuracy'])
        model.summary()

        # Model training
        rmse_train =  self.train_on_dataset(train_dataset_array, model)
        rmse_test =  self.test_on_dataset(test_dataset_array, model)

        # Plotting
        plt.plot(rmse_train, label='Train')
        plt.plot(rmse_test, label='Test')
        plt.xlabel('samples')
        plt.ylabel('error')
        plt.legend(loc='upper right')
        plt.show()
