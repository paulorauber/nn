import numpy as np
import time

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state

from scipy.special import expit


def sigmoid_prime(z):
    p = expit(z)
    return p*(1 - p)

sigmoid = expit


class FeedForwardNetwork(BaseEstimator, ClassifierMixin):

    def __init__(self, n_hidden_neurons, learning_rate=3.0, batch_size=10,
                 n_epochs=30, lmbda=5.0, output_activation='softmax',
                 random_state=None, verbose=0):
        self.n_hidden_neurons = n_hidden_neurons
        self.random_state = random_state

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lmbda = lmbda
        self.output_activation = output_activation

        self.verbose = verbose

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self.random_state_ = check_random_state(self.random_state)

        self.classes_, y = np.unique(y, return_inverse=True)
        t = np.zeros((y.shape[0], len(self.classes_)))
        for i, yi in np.ndenumerate(y):
            t[i, yi] = 1

        self.n_neurons_ = [X.shape[1]] + self.n_hidden_neurons +\
            [len(self.classes_)]

        self.n_layers_ = len(self.n_neurons_)

        self.W_ = []
        self.b_ = []
        for i in range(1, self.n_layers_):
            sdev = 1.0 / np.sqrt(self.n_neurons_[i - 1])
            dim = (self.n_neurons_[i], self.n_neurons_[i - 1])

            self.W_.append(self.random_state_.normal(0, sdev, size=dim))
            self.b_.append(self.random_state_.randn(self.n_neurons_[i]))

        self.stochastic_gradient_descent(X, t)

        return self

    def stochastic_gradient_descent(self, X, t):
        bs = self.batch_size

        now = time.time()
        for epoch in range(self.n_epochs):
            s = self.random_state_.permutation(X.shape[0])
            X_s, target_s = X[s], t[s]

            for i in range(0, X_s.shape[0], bs):
                self.gradient_descent_step(X_s[i: i + bs], target_s[i: i + bs],
                                           X.shape[0])

            if self.verbose > 0:
                now, last = time.time(), now
                print('Epoch {0} ({1:.01f}s).'.format(epoch + 1, now - last))

    def gradient_descent_step(self, X, t, n):
        # Computing current partial derivatives of cost wrt parameters
        partial_b = [np.zeros(b.shape, dtype=float) for b in self.b_]
        partial_W = [np.zeros(W.shape, dtype=float) for W in self.W_]

        for i in range(X.shape[0]):
            xi, ti = X[i], t[i]

            z, a = self.forward_pass(xi)
            err = [(a[-1] - ti)]

            for l in range(2, self.n_layers_):
                e = self.W_[-l + 1].T.dot(err[0]) * sigmoid_prime(z[-l])
                err.insert(0, e)

            for l in range(0, self.n_layers_ - 1):
                partial_b[l] += err[l]

                # Note that a[0] = xi
                partial_W[l] += err[l].reshape(-1, 1).dot(a[l].reshape(1, -1))

        decay = (1. - (self.learning_rate * self.lmbda) / n)
        # Moving parameters in opposite direction to the gradient
        for l in range(0, self.n_layers_ - 1):
            partial_b[l] /= X.shape[0]
            self.b_[l] -= self.learning_rate * partial_b[l]

            partial_W[l] /= X.shape[0]
            self.W_[l] = decay * self.W_[l] - self.learning_rate*partial_W[l]

    def forward_pass(self, x):
        z, a = [], [x]

        L = self.n_layers_ - 2
        for l in range(L):
            z.append(self.W_[l].dot(a[-1]) + self.b_[l])
            a.append(sigmoid(z[-1]))

        z.append(self.W_[L].dot(a[-1]) + self.b_[L])
        if self.output_activation == 'sigmoid':
            a.append(sigmoid(z[-1]))
        elif self.output_activation == 'softmax':
            Z = np.exp(z[-1])
            a.append(Z/Z.sum())

        return z, a

    def predict_proba(self, X):
        X = np.asarray(X)
        prob = np.zeros((X.shape[0], len(self.classes_)), dtype=float)

        for i, x in enumerate(X):
            a_L = self.forward_pass(x)[1][-1]
            prob[i] = a_L/a_L.sum()

        return prob

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
