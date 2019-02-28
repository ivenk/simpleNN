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
            prediction = self.think(training_set_input)
            error = training_set_result - prediction
            #adjust weights
            adjustment = dot(training_set_input.T, error * self.__sigmoid_derivative(prediction))
            self.weights += adjustment

    # think takes multiple training sets in form of a single matrix or just a single vector
    def think(self, input):
        return self.__sigmoid(dot(input, self.weights))


if __name__ == "__main__":
    iterations = 1000
    training_sets = array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
    training_sets_results = array([[0, 1, 1, 0]]).T
    data = array([1, 0, 0])
    
    print("Creating neural network")
    neuralnetwork = NeuralNetwork()
    print("Starting weights are:\n%s" % neuralnetwork.weights)
    
    print("starting training with %d iterations" % iterations)
    neuralnetwork.train(training_sets, training_sets_results, iterations)
    print("training completed!")
    print("New weights are:\n%s" % neuralnetwork.weights)

    print("Thinking about new problem:\n%s" % data)
    result = neuralnetwork.think(data)
    print("Result: %s" % result)
    


