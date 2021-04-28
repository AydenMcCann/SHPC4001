import numpy as np
import matplotlib.pyplot as plt


class DenseLayer():

    def __init__(self, inputLength, N, f):
        self.f = f
        limit = np.sqrt(6 / (inputLength + N))
        self.weights = np.random.uniform(-limit, limit, size=(N, inputLength))
        self.biases = np.random.uniform(-limit, limit, size=(N, 1))

        self.yOut = np.empty((N, 1))
        self.z = np.zeros((N, 1))

        self.delta = np.zeros((N, 1))
        self.weightsGradient = np.zeros((N, inputLength))
        self.deltaSum = np.zeros((N, 1))

    def input(self, yIn):
        self.z[:] = np.dot(self.weights, yIn) + self.biases
        self.yOut[:] = self.f(self.z)
        return self.yOut


class neuralNetwork():

    def __init__(self, layers):

        self.layers = layers

    def forwardPass(self, xs):
        yOut = xs
        for layer in self.layers:
            yOut = layer.input(yOut)
        return yOut

    def updateTheta(self, learningRate, batchSize):
        for layer in self.layers:
            layer.weights -= learningRate * layer.weightsGradient / batchSize
            layer.biases -= learningRate * layer.deltaSum / batchSize
            layer.weightsGradient[:] = 0
            layer.deltaSum[:] = 0

    def costFunction(self, yNetwork, yExpected):
        return np.sum((yNetwork - yExpected) ** 2)

    def validate(self, validationData):

        xValidate = validationData[0]
        yValidate = validationData[1]
        nSamples = len(xValidate)

        cost = 0

        for i in range(nSamples):
            yNetwork = self.forwardPass(xValidate[i][:, :])
            yExpected = yValidate[i][:, :]
            cost += self.costFunction(yNetwork, yExpected)

        return cost / nSamples

    def train(self, trainingData, validationData, batchSize, epochs, learningRate):

        # An 'epoch' is a complete pass through the training set.

        xTraining = trainingData[0]
        yTraining = trainingData[1]

        # nBatches = len(xTraining) // batchSize

        trainingPlot = []
        validationPlot = []
        batchPlot = []

        nBatch = 0
        #
        indexes = np.arange(len(xTraining))

        breakloop = 0
        for epoch in range(epochs + 1):
            epochComplete = False
            currentIndex = 0
            # np.random.shuffle(indexes)

            while not epochComplete:

                validationCost = 0
                trainingCost = 0

                for i in range(batchSize):


                    dataIndex = indexes[currentIndex]
                    yNetwork = self.forwardPass(xTraining[dataIndex])
                    yExpected = yTraining[dataIndex]
                    self.backPropagation(xTraining[dataIndex], yNetwork, yExpected)
                    self.updateTheta(learningRate, batchSize)
                    trainingCost += self.costFunction(yNetwork, yExpected)

                    currentIndex += 1
                    if currentIndex == len(xTraining):
                        epochComplete = True
                        break

                    trainingCost /= i + 1
                    validationCost = self.validate(validationData)

                nBatch += 1
                trainingPlot.append(trainingCost)
                validationPlot.append(validationCost)
                batchPlot.append(nBatch)
                print("Training Cost: ", trainingCost, "Validation Cost: ", validationCost)

            # prematurely ends the training if the validation cost gradient == 0
            if epoch > 1:
                if validationPlot[-1] == validationPlot[-2]:
                    print("Training ended after {} epochs due to a validation cost gradient of 0".format(epoch))
                    breakloop = 1
            if breakloop == 1:
                break

        plt.clf()
        plt.plot(batchPlot, trainingPlot, color='orange', label="training")
        plt.plot(batchPlot, validationPlot, color='green', label="validation")
        plt.legend()

    def backPropagation(self, x, yNetwork, yExpected):
        # Performs calculations for Nth layer
        zN = self.layers[-1].z
        fPrimeN = self.layers[-1].f(zN, derivative=True)
        deltaN = 2 * np.sum((yNetwork - yExpected) * fPrimeN)
        self.layers[-1].weightsGradient += np.outer(deltaN, self.layers[-2].yOut)
        self.layers[-1].deltaSum += deltaN

        # Loop from the 0th layer to the N-1th layer generating Z and fPrimes
        glo = globals()
        for i in range(1, len(layers)):
            glo["z%s" % i] = self.layers[i - 1].z
            glo["fPrime%s" % i] = self.layers[i-1].f(glo["z%s" % i], derivative=True)

        # Loop calculating delta values (working backwards due to their dependence on the previous layers delta)
        for i in range(len(layers)-1, 0, -1):
            if i == len(layers)-1:
                glo["delta%s" % i] = glo["fPrime%s" % i] * np.dot(np.transpose(self.layers[-1].weights), deltaN)
            else:
                glo["delta%s" % i] = glo["fPrime%s" % i] * np.dot(np.transpose(self.layers[i].weights), glo["delta%s" % (i + 1)])

        # Calculating remaining values for the first layer using the delta1 calculated above
        self.layers[0].weightsGradient += np.outer(delta1, x)
        self.layers[0].deltaSum += delta1

        # Calculating all other remaining deltaSums and weightsGradients
        for i in range(1, len(layers) - 1):
            self.layers[i].weightsGradient += np.outer(glo["delta%s" % (i + 1)], self.layers[i - 1].yOut)
            self.layers[i].deltaSum += glo["delta%s" % (i + 1)]


def identity(z, derivative=False):
    # An example activation function, it can be passed to the "Dense" class and then
    # called with "derivative = True" during backpropagation. You'll need to define
    # your own activation functions. Different layers can use different activation
    # functions.
    if derivative:
        return np.ones(z.shape)
    else:
        return z


def relu(z, derivative=False):
    if derivative:
        return np.where(z >= 0, 1, 0)
    else:
        return np.clip(z, 0, np.inf)


def sigmoid(z, derivative=False):
    if derivative:
        return sigmoid(z) * (1 - sigmoid(z))
    else:
        return 1 / (1 + np.exp(-z))


totalSamples = 4

# Arrays to contain (X, Y).
# Note that each of the x and y samples are defined as having two dimensions
# even though they are both 'vectors'. This is required to maintain dimensional
# consistency (from NumPy's perspective) over all required matrix operations.
xShape = (2, 1)
yShape = (1, 1)
xs = np.empty((totalSamples, xShape[0], xShape[1]))
ys = np.empty((totalSamples, yShape[0], yShape[1]))

xs[0, 0, 0] = 0
xs[0, 1, 0] = 0

xs[1, 0, 0] = 1
xs[1, 1, 0] = 0

xs[2, 0, 0] = 0
xs[2, 1, 0] = 1

xs[3, 0, 0] = 1
xs[3, 1, 0] = 1

ys[0, 0, 0] = 0
ys[1, 0, 0] = 1
ys[2, 0, 0] = 1
ys[3, 0, 0] = 0

trainingData = [xs[:, :, :], ys[:, :, :]]
validationData = [xs[:, :, :], ys[:, :, :]]

# dense1 and dense1 define the layers for an ANN with an input, hidden and output layer.
# The first argument gives the input size, the second argument is the number of neurons
# in the layer and the final argument is the activation function assigned to that layer.
# There is no layer object for the input layer as it is equal to the input data.
# Note that the number of neurons in the first layer dictates the input size of the
# next layer.


dense1 = DenseLayer(2, 2, sigmoid)
dense2 = DenseLayer(2, 1, identity)

layers = [dense1, dense2]

batchSize = 4
epochs = 10000
learningRate = 0.3

myNetwork = neuralNetwork(layers)
myNetwork.train(trainingData, validationData, batchSize, epochs, learningRate)

for x, y in zip(validationData[0], validationData[1]):
    print("y expected: ", y, "y Network:", myNetwork.forwardPass(x))

plt.show(block=True)
