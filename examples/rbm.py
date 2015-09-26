import gzip
import pickle
import numpy as np
from nn.model.rbm import RestrictedBoltzmannMachine

def print_digit(xi):
    x = xi.reshape(28, 28)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i, j]:
                print('#'),
            else:
                print('-'),
        print('')
    
    print("")

def main():
    np.random.seed(0)
    
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    train_Xy, _, _ = pickle.load(f)
    f.close()
    
    binary_threshold = 0.2
    X = train_Xy[0] > binary_threshold
    
    rbm = RestrictedBoltzmannMachine(n_hidden_variables=500, 
                                     learning_rate=0.01, batch_size=20,
                                     n_epochs=50, mu=0.5, pcd_steps=15,
                                     random_state=0, verbose=4)
          
    rbm.fit(X)
    
    n_restarts = 20
    for i in range(n_restarts):
        ind = np.random.randint(0, X.shape[0])
        print('Restart {0}. Initial image:'.format(i + 1))
        print_digit(X[ind])    
        
        Xs = rbm.sample(X[ind], sample_size=20, thinning=1000)
        
        for j, xs in enumerate(Xs):
            print('Restart {0}. Image {1}:'.format(i + 1, j + 1))
            print_digit(xs)
    
    
if __name__ == "__main__":
    main()