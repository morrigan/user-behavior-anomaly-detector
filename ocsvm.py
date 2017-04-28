import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
import numpy as np
import pandas
import helpers
from keras.preprocessing import sequence

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
trainX = sequence.pad_sequences(train_dataset_array, maxlen=max_vector_length, padding='pre')
testX = sequence.pad_sequences(test_dataset_array, maxlen=max_vector_length, padding='pre')

assert (trainX.shape[1] == testX.shape[1])

# fit the model
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(trainX)
y_pred_train = clf.predict(trainX)
y_pred_test = clf.predict(testX)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size

# plot the line, the points, and the nearest vectors to the plane
Z_vectors = yy.ravel()
for i in range(max_vector_length-1):
    Z_vectors = np.c_[Z_vectors,xx.ravel()]
Z = clf.decision_function(Z_vectors)
Z = Z.reshape(xx.shape)

print y_pred_train

plt.title("Novelty Detection")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

s = 40
b1 = plt.scatter(trainX[:, 0], trainX[:, 1], c='white', s=s)
b2 = plt.scatter(testX[:, 0], testX[:, 1], c='blueviolet', s=s)
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([a.collections[0], b1, b2],
           ["learned frontier", "training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left",
           prop=matplotlib.font_manager.FontProperties(size=11))
plt.xlabel(
    "Anomalies in training set: %d/%d; Anomalies in test set: %d/%d;"
    % (n_error_train, trainX.shape[0], n_error_test, testX.shape[0]))
plt.show()