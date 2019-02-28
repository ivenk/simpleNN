from numpy import array, exp, random, dot

class NeuralNetwork:
    def __init__(self):
        random.seed(1)
        # random.random(size) gives array of size
        # 2* and -1 shifts the range to (1, -1)
        self.weights = 2 * random.random((3,1)) - 1

    def __sigmoid(self,x):
        return 1/(1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1-x)

    def train(self, training_set_input, training_set_result, number_of_iterations):
        for i in range(0, number_of_iterations):
            prediction = self.think(training_set_inputs)
            error = training_set_result - prediction
            #adjust weights
            adjustment = dot(training_set_input.T, error * self.__sigmoid_derivative())
            self.weights += adjustment
            
    def think(self, input):
        return self.__sigmoid(dot(input.T * self.weights))
