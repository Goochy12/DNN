import numpy as np

np.random.seed(0)

# inputs layer

# hidden layers

# output layer


def dot(x, y):
    if not x:
        return 0
    return x[0] * y[0] + dot(x[1:], y[1:])


class layer:
    def __init__(self, inputs, neurons):
        self.weights = np.random.randn(inputs, neurons)
        self.biases = np.zeros((1, neurons))

    def foward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class dnn:
    def __init__(self, inputs):
        self.inputs = inputs
        return


X = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 1]])

weights = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6],
                    [0.7, 0.8, 0.9], [0.7, 0.8, 0.9]])
bias = 0

if __name__ == "__main__":
    l = layer(4, 4)
