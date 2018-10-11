###
# This program implements a neural network which predicts the output of 2-input XOR gate 
#
###
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np

train_X = np.array([[0, 0, 1],
                    [0, 1, 1],
                    [1, 0, 1],
                    [0, 1, 0],
                    [1, 0, 0],
                    [1, 1, 1],
                    [0, 0, 0]])

target = np.array([[0, 1, 1, 1, 1, 0, 0]]).T

model = Sequential()

model.add(Dense(30, activation='tanh', input_shape=(3,)))

model.add(Dense(20, activation='tanh'))

model.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.1)
model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

model_output = model.fit(train_X, target, epochs=10000)

test_X = np.array([[1.00, 1.00, 0.00], [0.998, 0.99, 0.00], [0.01, 0.95, 0.01], [0.98, 0.01, 0.02], [1.00, 0.00, 0.00], [0.12, 0.07, 0.00]])
predictions = model.predict_proba(test_X)

print('\nTesting on test data..\n')

for i in range(6):
    print('Predicted output for ', test_X[i], ' = ', np.round_(predictions[i], decimals=0))