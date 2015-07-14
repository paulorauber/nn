import numpy as np
import gzip
import pickle

from model.feed_forward import FeedForwardNetwork


def classify_mnist():
    np.set_printoptions(precision=4, suppress=True)

    f = gzip.open('data/mnist.pkl.gz', 'rb')
    train_Xy, valid_Xy, test_Xy = pickle.load(f)
    f.close()

    nn = FeedForwardNetwork([64, 32, 16], n_epochs=8, learning_rate=0.5,
                            random_state=0, batch_size=8, lmbda=8.0,
                            output_activation='softmax', verbose=1)

    X, y = train_Xy

    nn.fit(X, y)

    Xval, yval = valid_Xy
    print('Validation set accuracy: {0:.3f}.'.format(nn.score(Xval, yval)))

    Xtest, ytest = test_Xy
    print('Test set accuracy: {0:.3f}.'.format(nn.score(Xtest, ytest)))


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

if __name__ == "__main__":
    classify_mnist()
