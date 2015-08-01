About
-----
Neural networks in python.

This code is intended mainly as proof of concept of the algorithms presented in 
[1, 2, 3]. The implementations are not particularly clear, efficient, well tested 
or numerically stable. We advise against using this software for nondidactic 
purposes.

This software is licensed under the MIT License. 

Features
--------
Models:
    Feedforward neural network (classifier)
        Arbitrary number of layers
        Cross-entropy and negative log-likelihood cost functions
        Backpropagation
        Stochastic gradient descent
        L2 regularization
        Momentum

        Compatible with python sklearn interface
    
    Recurrent neural network (sequence element classifier)
        Single hidden layer
        Cross-entropy and negative log-likelihood cost functions
        Backpropagation
        Online gradient descent
        L2 regularization
        Momentum
    
Examples
--------
See the examples directory. 

Some examples use keras, a neural networks library.

References
----------
[1] Hinton, Geoffrey. Neural Networks for Machine Learning. Available in 
    http://www.coursera.org, 2015.
[2]  Nielsen, Michael. Neural Networks and Deep Learning.
    Available in http://neuralnetworksanddeeplearning.com, 2015.
[3] Graves, Alex. Supervised Sequence Labelling with Recurrent Neural Networks,
    2012.
