import numpy as np
import os
import csv

np.random.seed(0)


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


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


inputs, results = getInputs(os.path.join(
    os.path.dirname(__file__), "iris_flowers.csv"))

layer1 = Layer_Dense(4, 3)
activation1 = Activation_ReLU()

layer1.forward(inputs)

# print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)
