import numpy as np

train_X = np.asarray([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

train_Y = np.asarray([[0], [0], [0], [1]])

w0 = np.random.randn()
w1 = np.random.randn()
w2 = np.random.randn()

W = np.asarray([w0, w1, w2]).reshape((1, 3))

alpha = 0.1
epochs = 1000
m = len(train_X)

print("\nTraining the perceptron on the training data..")

for i in range(epochs):
    pred_Y = np.dot(train_X, np.transpose(W))

    for i in range(m):
        pred_Y[i] = 1.0 / (1 + (np.exp(pred_Y[i])))

    error = pred_Y - train_Y

    temp = alpha * (1.0/m) * np.dot(np.transpose(train_X), error)
    W = W + np.transpose(temp)

(w0, w1, w2) = (W[0][0], W[0][1], W[0][2])

print("\nTrained weights:")
print("w0 = ", w0, "w1 = ", w1, "w2 = ", w2)

test_X = [[0.82, 0.9], [0.47, 0.95], [0.73, 0.91], [0.59, 1.02], [0.89, 0.10], [0.932, 1.123], [0.85, 1.02], [0.12, 0.07], [0.205, 0.999], [0.13, 0.21], [0.73, 1.8], [0.22, 1.12], [0.77, 0.83], [0.1, 1.5], [0.4, 0.07]]
test_Y = [1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0] 

num_of_correct = 0

print("\nTesting the perceptron..")

for i in range(len(test_X)):
    temp = w0 + w1 * test_X[i][0] + w2 * test_X[i][1]

    temp = 1.0 / (1 + (np.exp(temp)))

    if (temp >= 0.5):
        pred = 1
    else:
        pred = 0

    print("\nx1: ", test_X[i][0], " x2: ", test_X[i][1], " Predicted: ", pred, " Correct: ", test_Y[i])

    if (pred == test_Y[i]):
        num_of_correct = num_of_correct + 1

print ("\nAccuracy: ", num_of_correct / len(test_Y))
