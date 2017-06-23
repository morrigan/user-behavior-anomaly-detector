#!/usr/bin/python

import math
import matplotlib.pyplot as plt
import numpy
import pandas
from keras.layers import LSTM as LSTM_CELL, Dense, Masking, Dropout, BatchNormalization
from keras.models import Sequential, model_from_json
from keras.utils import plot_model
from sklearn.metrics import mean_squared_error

import helpers

class LSTM:
    def __init__(self, config_file):
        self.settings = helpers.getConfig(config_file)

    def visualize_model_training(self, history):
        # summarize history for accuracy
        plt.plot(history.history['acc'], label="Train")
        if ('val_acc' in history.history):
            plt.plot(history.history['val_acc'], label="Test")
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(loc='upper left')
        plt.show()

        # summarize history for loss
        plt.plot(history.history['loss'], label="Train")
        if ('val_loss' in history.history):
            plt.plot(history.history['val_loss'], label="Test")
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(loc='upper left')
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
        #model.add(Masking(mask_value=0, input_shape=(1, self.settings.getint("LSTM", "max_vector_length"))))
        model.add(LSTM_CELL(self.settings.getint("LSTM", "hidden_layers"),
                            input_shape=(self.settings.getint("LSTM", "time_series"), self.settings.getint("LSTM", "max_vector_length")),
                            return_sequences=True))
        model.add(LSTM_CELL(self.settings.getint("LSTM", "hidden_layers")))
        model.add(Dropout(self.settings.getfloat("LSTM", "dropout")))
        model.add(Dense(self.settings.getint('LSTM', 'max_vector_length')))

        return model


    def update_model(self, model, x, y):
        return self.train_model(model, x, y, verbose=0)


    def get_model(self):
        # LSTM model
        if (self.settings.getboolean("LSTM", "load_existing_model") == True):
            model = self.load_model()
        else:
            model = self.create_model()
        model.compile(loss="mean_squared_error", optimizer="adam", metrics=['accuracy'])
        model.summary()

        return model


    def train_model(self, model, trainX, trainY, validation_data = None, verbose=2):
        return model.fit(trainX, trainY,
                         validation_data=validation_data,
                         epochs=self.settings.getint("LSTM", "epochs"),
                         batch_size=self.settings.getint("LSTM", "batch_size"), verbose=verbose, shuffle=False)


    def calculate_score(self, real, predicted, score):
        rmse = math.sqrt(mean_squared_error(real, predicted))
        # total = helpers.sigmoid(1/rmse * scores[i]) * 100
        total = round(rmse * score / 100, 2)
        return total


    def forecast(self, model, x, y):
        x = numpy.reshape(x, (x.shape[0], 1, x.shape[1]))
        predicted = model.predict(x, batch_size=1, verbose=0)

        # Update model with new actions
        self.update_model(model, x, y)   # save data

        return predicted


    def load_datasets(self):
        # Create datasets
        train_dataset = pandas.read_csv(self.settings.get('Data', 'train_dataset_file'), delimiter=';', engine='python')
        test_dataset = pandas.read_csv(self.settings.get('Data', 'test_dataset_file'), delimiter=';', engine='python')

        # Convert strings
        train_dataset_array = helpers.collection_values_to_array(train_dataset)
        test_dataset_array = helpers.collection_values_to_array(test_dataset)  # shape (n,)

        return train_dataset_array, test_dataset_array


    def pretransform_dataset(self, dataset, reshape = False):
        max_vector_length = self.settings.getint("LSTM", "max_vector_length")

        dataset_padded = helpers.padding(dataset, max_vector_length)

        assert (dataset_padded.shape[1] == max_vector_length)

        if (self.settings.getboolean("LSTM", "scale_data") == True):
            if (reshape == True):
                dataset_padded = dataset_padded.reshape(-1, 1)

            self.scaler, datasetX = helpers.scale(dataset_padded)

            if (reshape == True):
                datasetX = datasetX.reshape(1, -1)
        else:
            datasetX = dataset_padded

        return datasetX


    def transform_dataset(self, dataset):
        datasetX = self.pretransform_dataset(dataset)

        # Normalize
        #helpers.normalize(datasetX)

        # For predicting
        datasetX, datasetY = helpers.convert_to_timeseries(datasetX, self.settings.getint("LSTM", "time_series"))

        return datasetX, datasetY


    """
        Train dataset on a given model (existing or created) and make predictions.
        Return root mean squared error.
    """
    def train_on_dataset(self, trainX, trainY, model):
        # Visualize model structure
        if (self.settings.getboolean("LSTM", "visualize_model") == True):
            plot_model(model, to_file='model.png')

        # Model training
        if (self.settings.getboolean("LSTM", "load_existing_model") == False):
            history = self.train_model(model, trainX, trainY)

            # Visualize model training
            if (self.settings.getboolean("LSTM", "visualize_model") == True):
                self.visualize_model_training(history)

            # Save model
            if (self.settings.getboolean("LSTM", "save_training_model") == True):
                self.save_model(model)

        # Make predictions
        trainPredict = model.predict(trainX)
        # trainPredict error
        rmse = helpers.calculate_rmse(trainX, trainPredict)

        return rmse


    """
        Use loaded model to run a test dataset on it.
        Return root mean squared error.
    """
    def test_on_dataset(self, testX, testY, model):
        # Make predictions
        testPredict = model.predict(testX)

        # Calculate error
        rmse = helpers.calculate_rmse(testX, testPredict)

        return rmse


    """
        Take both train and validation test through fitting to get
        validation set accuracy.
    """
    def train_validate_dataset(self, trainX, trainY, testX, testY, model):
        # Visualize model structure
        if (self.settings.getboolean("LSTM", "visualize_model") == True):
            plot_model(model, to_file='model.png')

        # Model training
        history = self.train_model(model, trainX, trainY, validation_data=(testX, testY))

        # Visualize model training
        if (self.settings.getboolean("LSTM", "visualize_model") == True):
            self.visualize_model_training(history)


    """
        Use preprocessed dataset file for training and testing.
        Use model related configuration from settings.ini file.
    """
    def run_on_datasets(self, train_dataset_array = [], test_dataset_array = []):
        # If none given, load datasets from .csv defined in settings
        if (len(train_dataset_array) == 0 and len(test_dataset_array) == 0):
            train_dataset_array, test_dataset_array = self.load_datasets()

        # Transform dataset
        trainX, trainY = self.transform_dataset(train_dataset_array)
        testX, testY = self.transform_dataset(test_dataset_array)

        self.input_shape = trainX.shape

        model = self.get_model()

        #self.train_validate_dataset(trainX, trainY, testX, testY, model)

        # Model training
        rmse_train =  self.train_on_dataset(trainX, trainY, model)
        rmse_test =  self.test_on_dataset(testX, testY, model)

        # Plotting
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.plot(rmse_train, label='Train')
        ax2.plot(rmse_test, label='Test')
        plt.xlabel('samples')
        plt.ylabel('error')
        ax1.legend(loc='upper right')
        ax2.legend(loc='upper right')
        plt.show()
