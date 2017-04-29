import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
import numpy as np
import pandas
import helpers
from keras.preprocessing import sequence
from sklearn import metrics

def print_accuracy(title, datasetY, predictions):
    print title

    print("accuracy: ", metrics.accuracy_score(datasetY, predictions))
    print("precision: ", metrics.precision_score(datasetY, predictions))
    print("recall: ", metrics.recall_score(datasetY, predictions))
    print("f1: ", metrics.f1_score(datasetY, predictions))
    print("area under curve (auc): ", metrics.roc_auc_score(datasetY, predictions))

xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
max_vector_length = 30

# Create datasets
train_dataset = pandas.read_csv("./data/test.csv", delimiter=';', engine='python')
test_dataset = pandas.read_csv("./data/primjer.csv", delimiter=';', engine='python')

train_dataset = train_dataset[:len(train_dataset)/12]

# Convert strings
train_dataset_array = helpers.collection_values_to_array(train_dataset)
test_dataset_array = helpers.collection_values_to_array(test_dataset)

# Padding (from left)
trainX = sequence.pad_sequences(train_dataset_array, maxlen=max_vector_length)
testX = sequence.pad_sequences(test_dataset_array, maxlen=max_vector_length) #padding='pre'

assert (trainX.shape[1] == testX.shape[1])

# fit the model
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(trainX)
y_pred_train = clf.predict(trainX)
y_pred_test = clf.predict(testX)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size

# Display accuracy on validation set
#print_accuracy("Validation", testX, y_pred_test)

# plot the line, the points, and the nearest vectors to the plane
#Z_vectors = yy.ravel()
#for i in range(max_vector_length-1):
#    Z_vectors = np.c_[Z_vectors,xx.ravel()]
#Z = clf.decision_function(Z_vectors)
#Z = Z.reshape(xx.shape)

#print Z
#print Z.shape

plt.title("Novelty Detection")
#plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
#a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
#plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')


plt.figure(1)
plt.subplot(211)
b1 = plt.plot(trainX, 'ro', testX, 'g^')
plt.subplot(212)
b1 = plt.plot(y_pred_train, 'ro', y_pred_test, 'g^')
plt.legend([b1],
           ["training observations",
            "test observations"],
           loc="upper left",
           prop=matplotlib.font_manager.FontProperties(size=11))
plt.xlabel(
    "Anomalies in training set: %d/%d; Anomalies in test set: %d/%d;"
    % (n_error_train, trainX.shape[0], n_error_test, testX.shape[0]))
plt.show()