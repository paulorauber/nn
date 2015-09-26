import numpy as np
import time
from sklearn.utils import check_random_state

from scipy.special import expit

def sigmoid(x):
    return expit(np.clip(x, -30, 30))

class RestrictedBoltzmannMachine:
    def __init__(self, n_hidden_variables, learning_rate=0.1, batch_size=20,
                 n_epochs=15, mu=0.5, pcd_steps=1, random_state=None, verbose=0):
        self.n_hidden = n_hidden_variables
        self.random_state = random_state

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.mu = mu
        self.pcd_steps = pcd_steps

        self.verbose = verbose
    
    def fit(self, X):
        self.random_state_ = check_random_state(self.random_state)
        X = np.asarray(X, dtype=np.bool)
        
        self.n_visible_ = X.shape[1]
        self.init_parameters()
        
        self.stochastic_gradient_descent(X)
        
        return self
        
    def init_parameters(self):
        sdev = 1.0 / np.sqrt(self.n_visible_)
        dim = (self.n_hidden, self.n_visible_)

        self.W_ = self.random_state_.normal(0, sdev, size=dim)
        self.b_ = np.zeros(self.n_visible_)
        self.c_ = np.zeros(self.n_hidden)
        
        self.VW_ = np.zeros(self.W_.shape)
        self.Vb_ = np.zeros(self.b_.shape)
        self.Vc_ = np.zeros(self.c_.shape)
        
        self.V = None
        
    def stochastic_gradient_descent(self, X):
        bs = self.batch_size

        now = time.time()
        for epoch in range(self.n_epochs):
            s = self.random_state_.permutation(X.shape[0])
            X_s = X[s]

            for i in range(0, X_s.shape[0], bs):
                self.gradient_descent_step(X_s[i: i + bs])
                
                if self.verbose > 2:
                    print('Epoch {0} ({1:.1f}%).'.format(epoch + 1, 
                          100*float(i)/(X_s.shape[0] - 1)))

            if self.verbose > 0:
                now, last = time.time(), now
                print('Epoch {0} ({1:.01f}s).'.format(epoch + 1, now - last))
                if self.verbose > 3:
                    print('Average reconstruction error: {0:.3f}.'.\
                        format(self.reconstruction_error(X[0: 10*bs])))
                
    def gradient_descent_step(self, X):
        if self.V is None:
            self.V = np.array(X, dtype=np.bool)
            
        for i, vi in enumerate(self.V):
            self.V[i] = self.sample(vi, 1, thinning=self.pcd_steps - 1)[0]
                
        neg_W = np.zeros(self.W_.shape)
        neg_b = np.zeros(self.b_.shape)
        neg_c = np.zeros(self.c_.shape)

        """Note: we skip the division by self.V.shape[0] = self.X.shape[0],
         since it would be cancelled by a multiplication by self.X.shape[0] 
         before updating the partial derivatives. """
        for vi in self.V:
            a = sigmoid(self.W_.dot(vi) + self.c_)
            
            neg_W += a.reshape(-1, 1)*vi.reshape(1, -1)
            neg_b += vi
            neg_c += a
            
        partial_W = np.zeros(self.W_.shape)
        partial_b = np.zeros(self.b_.shape)
        partial_c = np.zeros(self.c_.shape)
        for xi in X:
            a = sigmoid(self.W_.dot(xi) + self.c_)
            
            partial_W += a.reshape(-1, 1)*xi.reshape(1, -1)
            partial_b += xi
            partial_c += a
            
        partial_W -= neg_W
        partial_b -= neg_b
        partial_c -= neg_c
            
        self.VW_ = self.mu*self.VW_ + self.learning_rate*partial_W
        self.Vb_ = self.mu*self.Vb_ + self.learning_rate*partial_b
        self.Vc_ = self.mu*self.Vc_ + self.learning_rate*partial_c
        
        self.W_ += self.VW_
        self.b_ += self.Vb_
        self.c_ += self.Vc_
        
    def sample(self, x, sample_size, thinning=0):
        v = np.array(x)
        samples = []
        
        for _ in range(sample_size):
            for _ in range(thinning + 1):
                a = sigmoid(self.W_.dot(v) + self.c_)
                thresholds = np.random.random(self.n_hidden)
                h = (a > thresholds)
                
                a = sigmoid(self.W_.T.dot(h) + self.b_)
                thresholds = np.random.random(self.n_visible_)
                v = (a > thresholds)
        
            samples.append(v)
        
        return np.array(samples, dtype=np.bool)
        
    def reconstruction_error(self, X):
        X = np.asarray(X, dtype=np.bool)
        
        e = 0.0
        for xi in X:
            vi = self.sample(xi, 1)[0]
            e += (xi != vi).sum()
        e /= X.size
        
        return e