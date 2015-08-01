import gzip
import pickle

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D

from keras.optimizers import SGD


def print_digit(xi, yi):
    x = xi.reshape(28, 28)

    print('Digit:')

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i, j] > 0.1:
                print('#'),
            else:
                print('-'),
        print('')

    print('Label: {0}'.format(yi))  

def create_model(load_weights=False):
    nn = Sequential()
    nn.add(Convolution2D(32, 1, 3, 3, border_mode='same', activation='relu'))
    nn.add(Convolution2D(32, 32, 3, 3, border_mode='same', activation='relu'))
    nn.add(MaxPooling2D(poolsize=(2,2)))
    nn.add(Dropout(0.25))
    
    nn.add(Convolution2D(64, 32, 3, 3, border_mode='same', activation='relu')) 
    nn.add(Convolution2D(64, 64, 3, 3, border_mode='same', activation='relu'))
    nn.add(MaxPooling2D(poolsize=(2,2)))
    nn.add(Dropout(0.25))
    
    nn.add(Flatten())
    nn.add(Dense(64*7*7, 256, activation='relu'))
    nn.add(Dropout(0.5))
    
    nn.add(Dense(256,10, activation='softmax'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    nn.compile(loss='categorical_crossentropy', optimizer=sgd)
    
    if load_weights:
        nn.load_weights('cnn_weights.hdf5')
    
    return nn
    
def train_model(datapath, nn, n_classes, n_samples):
    f = gzip.open(datapath, 'rb')
    train_Xy, _, _ = pickle.load(f)
    f.close()

    # y[i] is assumed to be in [0, n_classes)
    X, y = train_Xy
    Y = np.zeros((y.shape[0], n_classes))
    for i, yi in np.ndenumerate(y):
        Y[i, yi] = 1
        
    # Subsampling. Input must be shuffled previously.
    X = X[:n_samples]
    Y = Y[:n_samples]
    
    X = X.reshape(X.shape[0], 1, 28, 28)
    nn.fit(X, Y, batch_size=32, nb_epoch=1, validation_split=0.1, 
           show_accuracy=True, verbose=1)
           
    return nn
    
def evaluate_model(datapath, nn, n_samples):
    f = gzip.open(datapath, 'rb')
    _, valid_Xy, _ = pickle.load(f)
    f.close()
    
    # yval[i] is assumed to be in [0, n_classes)
    Xval, yval = valid_Xy
    
    # Subsampling. Input must be shuffled previously.
    Xval = Xval[:n_samples]
    yval = yval[:n_samples]

    Xval = Xval.reshape(Xval.shape[0], 1, 28, 28)    
    ypred = nn.predict_classes(Xval, batch_size=32, verbose=1)
    
    score = (ypred == yval).sum() / float(ypred.shape[0])

    print('Validation set accuracy: {0:.3f}.'.format(score))

def main():
    datapath = 'data/mnist.pkl.gz'
    
    nn = create_model(load_weights=False)
    
    nn = train_model(datapath, nn, n_classes=10, n_samples=50000)
    nn.save_weights('cnn_weights.hdf5', overwrite=True)
    
    evaluate_model(datapath, nn, n_samples=10000)
    
if __name__ == "__main__":
    main()