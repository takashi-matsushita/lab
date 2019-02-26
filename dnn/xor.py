### train DNN for XOR
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

import matplotlib.pyplot as plt
plt.style.use('bmh') 


### set seeds for reproduciability
import random; random.seed(1111)
import numpy as np; np.random.seed(3333)
import tensorflow as tf; tf.set_random_seed(5555)

### XOR training data
X = np.array([[0,0],[0,1],[1,0],[1,1]])  # input
y = np.array([[0],[1],[1],[0]])          # output


### MLP with two input variables (except for bias),
### a hidden layer with 2 perceptrons and output layer with 1 perceptron
model = Sequential()
model.add(Dense(2, input_dim=2, use_bias=True))  # hidden layer
model.add(Activation('tanh'))                    # hyperbolic tangent activation function
model.add(Dense(1, use_bias=True))               # output layer
model.add(Activation('sigmoid'))                 # sigmoid activation function

### training
sgd = SGD(lr=0.1)   # stochastic gradient descent optimiser with learning rate = 0.1
model.compile(loss='binary_crossentropy', optimizer=sgd)
hist = model.fit(X, y, batch_size=1, epochs=2000, verbose=1)

### plot loss vs epoch
plt.plot(hist.history['loss'], label='loss')
plt.yscale('log')
plt.xlabel('epoch')
plt.legend()
plt.show()

### check prediction
print(model.predict(X))
threshold = 0.5
print((model.predict(X) > threshold).astype(np.int))

# eof
