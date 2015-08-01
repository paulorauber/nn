import gzip
import pickle

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

def feed_forward():
    # Loading data
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    train_Xy, valid_Xy, test_Xy = pickle.load(f)
    f.close()
    
    X, y = train_Xy
    classes, y = np.unique(y, return_inverse=True)
    Y = np.zeros((y.shape[0], len(classes)))
    for i, yi in np.ndenumerate(y):
        Y[i, yi] = 1
    
    # Model definition
    nn = Sequential()
    nn.add(Dense(X.shape[1], 300, init='he_uniform', activation='relu'))
    nn.add(Dense(300, 300, init='he_uniform', activation='relu'))
    nn.add(Dense(300, 10, init='he_uniform', activation='softmax'))
    
    sgd = SGD(lr=0.1, decay=1e-9, momentum=0.5, nesterov=True)
    nn.compile(loss='categorical_crossentropy', optimizer=sgd)
    
    # Training
    nn.fit(X, Y, nb_epoch=20, batch_size=16, verbose=2)
    
    # Evaluating
    Xval, yval = valid_Xy
    classes = {v: i for (i, v) in enumerate(classes)}
    yval = np.array([classes[yi] for yi in yval], dtype=int)
    ypred = nn.predict_classes(Xval, verbose=0)
    
    score = (ypred == yval).sum() / float(ypred.shape[0])

    print('Validation set accuracy: {0:.3f}.'.format(score))
    

if __name__ == "__main__":
    feed_forward()