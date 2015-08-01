import numpy as np
import gzip
import pickle

from model.feed_forward import FeedForwardNetwork
from model.feed_forward import FeedForwardNetworkMomentum

def classify_mnist(nn):
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    train_Xy, valid_Xy, test_Xy = pickle.load(f)
    f.close()

    X, y = train_Xy
    
    nn.fit(X, y)

    Xval, yval = valid_Xy
    print('Validation set accuracy: {0:.3f}.'.format(nn.score(Xval, yval)))


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
    
    
def main():
    np.set_printoptions(precision=4, suppress=True)

    nn = FeedForwardNetwork([30], n_epochs=5, learning_rate=0.5,
                            random_state=0, batch_size=10, lmbda=5.0,
                            output_activation='softmax', verbose=1)
                            
    classify_mnist(nn)

    nn = FeedForwardNetworkMomentum([64, 32, 16], n_epochs=5, learning_rate=0.1,
                            random_state=0, batch_size=10, lmbda=5.0, mu=0.5,
                            output_activation='softmax', verbose=1)
                            
    classify_mnist(nn)

if __name__ == "__main__":
    main()
