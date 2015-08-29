import numpy as np
from sklearn.utils import check_random_state

from nn.model.lstm import LongShortTermMemoryNetwork

random_state = check_random_state(None)

def nback(n, k, length):
    """Random n-back targets given n, number of digits k and sequence length"""
    Xi = random_state.randint(k, size=length)
    yi = np.zeros(length, dtype=int)
    
    for t in range(n, length):
        yi[t] = (Xi[t - n] == Xi[t]) 
        
    return Xi, yi
    
def one_of_k(Xi_, k):
    Xi = np.zeros((len(Xi_), k))
    for t, Xit in np.ndenumerate(Xi_):
        Xi[t, Xit] = 1    
        
    return Xi
    
def nback_dataset(n_sequences, mean_length, std_length, n, k):
    X, y = [], []
    
    for _ in range(n_sequences):
        length = random_state.normal(loc=mean_length, scale=std_length)
        length = int(max(n + 1, length))
        
        Xi_, yi = nback(n, k, length)
        Xi = one_of_k(Xi_, k)
            
        X.append(Xi)
        y.append(yi)
        
    return X, y
    
def nback_example():
    # Input dimension
    k = 4
    # n-back
    n = 3
    
    n_sequences = 100
    mean_length = 20
    std_length = 5
    
    # Training 
    Xtrain, ytrain = nback_dataset(n_sequences, mean_length, std_length, n, k)
    
    lstm = LongShortTermMemoryNetwork(64, learning_rate=2.0, n_epochs=70,
                                      mu=0.7, output_activation='softmax', 
                                      random_state=None, verbose=1)
                 
    lstm.fit(Xtrain, ytrain)
    
    # Evaluating
    Xtest, ytest = nback_dataset(5*n_sequences, 5*mean_length, 5*std_length, 
                                 n, k)
    
    print('\nTest score: {0:.3f}.'.format(lstm.score(Xtest, ytest)))
    
    acc_zeros = 0.0
    for yi in ytest:
        acc_zeros += float((yi == 0).sum()) / len(yi)
    acc_zeros /= len(ytest)
    print('Negative guess score: {0:.3f}.'.format(acc_zeros))
    
    # Example
    Xi_ = [3, 2, 1, 3, 2, 1, 3, 2, 2, 1, 2, 3, 1, 2, 0, 0, 2, 0]
    print('\nExample sequence: {0}.'.format(Xi_))
    yi = np.zeros(len(Xi_), dtype=int)
    for t in range(n, len(Xi_)):
        yi[t] = (Xi_[t - n] == Xi_[t]) 
        
    Xi = one_of_k(Xi_, k)
        
    yipred = lstm.predict([Xi])[0]
    print('Correct: \t{0}.'.format(yi))
    print('Predicted: \t{0}.'.format(yipred))
    print('Score: {0:.3f}.'.format(float((yi == yipred).sum())/len(yi)))

def main():
    nback_example()

if __name__ == "__main__":
    main()