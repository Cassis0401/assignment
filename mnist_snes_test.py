from timeit import default_timer as timer
import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from keras_helper import NNWeightHelper
from keras.utils import np_utils
from snes import SNES
import pickle as pkl

# use just a small sample of the train set to test
SAMPLE_SIZE = 1024
# how many different sets of weights ask() should return for evaluation
POPULATION_SIZE = 10
# how many times we will loop over ask()/tell()
GENERATIONS = 30

def train_classifier(model, X, y):
    X_features = model.predict(X)
    clf = RandomForestClassifier()
    clf.fit(X_features, y)
    y_pred = clf.predict(X_features)
    return clf, y_pred
def predict_classifier(model, clf, X):
    X_features = model.predict(X)
    return clf.predict(X_features)
# input image dimensions
img_rows, img_cols = 28, 28
num_classes = 10

# Load MNIST dataset from keras for source domain
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1).astype(np.uint8) * 255
x_test = x_test.reshape(10000, 28, 28, 1).astype(np.uint8) * 255
x_train1 = x_train.reshape(60000, 28, 28, 1).astype(np.uint8) * 255
x_test1 = x_test.reshape(10000, 28, 28, 1).astype(np.uint8) * 255
x_train = np.concatenate((x_train,x_test1 ), axis =0)
x_train = np.concatenate([x_train, x_train, x_train], 3)
x_train1 = np.concatenate([x_train1, x_train1, x_train1], 3)
x_test1 = np.concatenate([x_test1, x_test1, x_test1], 3)
y_train = np.concatenate((y_train,y_test), axis =0)

# Load MNIST-M dataset for target domain
mnistm = pkl.load(open('mnistm_data.pkl', 'rb'))
mnistm_train = mnistm['train']
mnistm_test = mnistm['test']
mnistm_valid = mnistm['valid']
x_test = np.concatenate((mnistm_train,mnistm_test, mnistm_valid), axis =0)
mnistm_train2=np.concatenate((mnistm_train,mnistm_valid), axis=0)
y_test = y_train

#domain data
x_train_domain = np.concatenate((x_train1,mnistm_train2), axis=0)
y_train_domain = np.concatenate((np.zeros(x_train1.shape[0]), np.ones(mnistm_train2.shape[0])),axis=0)
# print(x_train_domain.shape)
# print(y_train_domain[59000])#firt 59000 samples assign 0.0 and last 60000 is 1.0= totaly 120000 sample
# print(y_train_domain[60000])
x_test_domain = np.concatenate((x_test1,mnistm_test), axis=0)# totaly 20000 sample
y_test_domain = np.concatenate((np.zeros(x_test1.shape[0]), np.ones(mnistm_test.shape[0])),axis=0)
#print(x_test_domain.shape)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#one hot encode outputs
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#bu hocanın ki
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='relu'))

# this is irrelevant for what we want to achieve
#model.compile(loss="mse", optimizer="adam")
# print("compilation is over")
nnw = NNWeightHelper(model)
weights = nnw.get_weights()

def main():
    print("Total number of weights to evolve is:", weights.shape)
    all_examples_indices = list(range(x_train.shape[0]))
    clf, _ = train_classifier(model, x_train, y_train)
    y_pred = predict_classifier(model, clf, x_test)
    print(y_test.shape, y_pred.shape)

    test_accuracy = accuracy_score(y_test, y_pred)
    print('Non-trained NN Test accuracy:', test_accuracy)
    # print('Test MSE:', test_mse)

    snes = SNES(weights, 1, POPULATION_SIZE)
    for i in range(0, GENERATIONS):
        start = timer()
        asked = snes.ask()

        # to be provided back to snes
        told = []

        # use a small number of training samples for speed purposes
        subsample_indices = np.random.choice(all_examples_indices, size=SAMPLE_SIZE, replace=False)
        # evaluate on another subset
        subsample_indices_valid = np.random.choice(all_examples_indices, size=SAMPLE_SIZE + 1, replace=False)

        # iterate over the population
        for asked_j in asked:
            # set nn weights
            nnw.set_weights(asked_j)
            # train the classifer and get back the predictions on the training data
            clf, _ = train_classifier(model, x_train[subsample_indices], y_train[subsample_indices])
            clf2, _ = train_classifier(model, x_train_domain[subsample_indices], y_train_domain[subsample_indices])

            # calculate the predictions on a different set
            y_pred = predict_classifier(model, clf, x_train[subsample_indices_valid])
            # score = accuracy_score(y_train[subsample_indices_valid], y_pred)
            loss_lab = mean_squared_error(y_train[subsample_indices_valid], y_pred)
            
            y_pred2 = predict_classifier(model, clf2, x_train_domain[subsample_indices_valid])
            score2 = accuracy_score(y_train_domain[subsample_indices_valid], y_pred2)
            loss_dom = mean_squared_error(y_train_domain[subsample_indices_valid], y_pred)
            # clf, _ = train_classifier(model, x_train, y_train)
            # y_pred = predict_classifier(model, clf, x_test)
            # score = accuracy_score(y_test, y_pred)
            # append to array of values that are to be returned
            #inverted = (-score2)
            total_loss = (loss_lab + (2*-loss_dom))# change this functions

            told.append(total)

        snes.tell(asked, told)
        end = timer()
        print("It took", end - start, "seconds to complete generation", i + 1)

    nnw.set_weights(snes.center)

    clf, _ = train_classifier(model, x_train, y_train)
    y_pred = predict_classifier(model, clf, x_test)


    print(y_test.shape, y_pred.shape)
    test_accuracy = accuracy_score(y_test, y_pred)


    print('Test accuracy:', test_accuracy)


if __name__ == '__main__':
    main()
