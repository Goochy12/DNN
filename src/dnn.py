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
    outputs = []
    next(reader)
    for eachLine in reader:
        values = []
        for eachValue in eachLine[:len(eachLine)-1]:
            values.append(float(eachValue))
        inputs.append(values)
        outputs.append(eachLine[len(eachLine)-1])
    f.close()
    return inputs, outputs


class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self):
        pass


class dnn:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.layers = []
        self.activation = []

    def addLayer(self, n_inputs, n_neurons):
        self.layers.append(Layer(n_inputs, n_neurons))

    def forward(self):
        for eachLayer in self.layers:
            eachLayer.forward(self.inputs)
            self.activation.append(Activation())
            self.activation[len(self.activation)-1].foward(eachLayer.output)


class Activation:
    def foward(self, inputs):
        self.output = self.relu(inputs)

    def relu(self, inputs):
        return np.maximum(0, inputs)


X = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 1]])

if __name__ == "__main__":
    inputs, outputs = getInputs(os.path.join(
        os.path.dirname(__file__), "iris_flowers.csv"))

    dnn = dnn(inputs, outputs)
    dnn.addLayer(len(inputs[0]), 3)
    dnn.addLayer(4, 3)
    dnn.forward()
