import numpy as np
import csv
import os

np.random.seed(0)

# inputs layer

# hidden layers

# output layer


def dot(x, y):
    if not x:
        return 0
    return x[0] * y[0] + dot(x[1:], y[1:])


def getInputs(filename):
    f = open(filename, 'r+')
    reader = csv.reader(f)
    inputs = []
    results = []
    next(reader)
    for eachLine in reader:
        values = []
        for eachValue in eachLine[:len(eachLine)-1]:
            values.append(float(eachValue))
        inputs.append(values)
        results.append(eachLine[len(eachLine)-1])
    f.close()
    return inputs[1:], results


class layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self):
        pass


class dnn:
    def __init__(self, inputs, results):
        self.inputs = inputs
        self.results = results
        self.layers = []
        return

    def addLayer(self, n_inputs, n_neurons):
        self.layers.append(layer(n_inputs, n_neurons))

    def forward(self):
        for eachLayer in self.layers:
            eachLayer.forward(self.inputs)
            self.inputs = self.activate(eachLayer.output)

    def activate(self, x):
        return self.relu(x)

    def relu(self, x):
        return np.maximum(0, x)


X = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 1]])

if __name__ == "__main__":
    inputs, results = getInputs(os.path.join(
        os.path.dirname(__file__), "iris_flowers.csv"))

    dnn = dnn(inputs, results)
    dnn.addLayer(len(inputs[0]), 4)
    dnn.addLayer(4, 3)
    dnn.forward()
    print(dnn.inputs)
