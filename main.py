from tensorflow import keras
import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.W1 = np.random.randn(784,128) * 0.01 # Weight initialziation
        self.W2 = np.random.randn(128, 10) * 0.01
        self.b1 = np.zeros((128, 1)) # Bias initialization
        self.b2 = np.zeros((10, 1))

    def sigmoid(self, arr):
        return 1/ (1+np.exp(-arr)) # Sigmoid function for compressing the weights into 0 - 1 range

    def forward_propagation(self, X):
        """
        X = 784 x 1 array representing the 28 x 28 pixel image
        W = Weight
        b = Bias
        """
        z1 = self.W1.T @ X + self.b1
        a1 = self.sigmoid(z1)
        z2 = self.W2.T @ a1 + self.b2
        a2 = self.sigmoid(z2)
        return a1, a2

    def calculate_loss(self, y_one_hot, prediction):
        """
        y_one_hot = an array of n zeros except the correct value inside of the image with a value of one where n is the number of possible predictions
        prediction = array of predictions from the model
        """
        return np.mean((prediction - y_one_hot) ** 2)
        
    
    def backwards_propagation(self, X, y_one_hot, a1, a2):
        """
        Backwards propagation will determine how to tweak the weights and biases so the model can learn from its mistakes
        a1, a2 = the activations of the neuron layers
        X = 784 x 1 array representing the 28 x 28 pixel image
        label = actual value presented in the image

        Derivative of our MSE Loss function (L = (1/n) * summation(a2 - y)^2) = (2/n) x (a2 - y)
        We will simply absorb the 2/n for now
        So the new derivative becomes a2 - y

        The chain rule will be used for the rest of the derivates, so in this case dL/dz2 = dL/da2 x da2/dz2
        Calculating da2/dz2 we will use the sigmoid derivative property which states if a = sigmoid(z) then:
        derivate = a * (1-a)
        Since we have dL/da2 alreadt calculate (see above) we will write this out as dL_da2 * a2 * (1 - a2)

        For dL/DW2 we know that z2 = W2.T @ a1 + b2, using the chain rule we can write the derivate as:
        dL/dW2 = dL/dz2 * dz2/dW2
        the derivative of z2 with respect to W2 will simply be a1 and we have already calculated dL_dz2 so it will be written:
        dl_DW2 = a1 @ dl_dz2.T

        Finally, dL_dz = dL_db because b is a constant and when derived will become equal to 1.
        """
        # Calculate derivate of weight 2
        dL_da2 = a2 - y_one_hot
        dL_dz2 = dL_da2 * a2 * (1 - a2) # doubles as dL_db2 because when derived b becomes 1
        dL_dW2 = a1 @ dL_dz2.T

        # Calculate derivative of weight 1
        dL_da1 = self.W2 @ dL_dz2
        dL_dz1 = dL_da1 * a1 * (1 - a1) # doubles as dL_db1 because when derived b becomes 1
        dL_dW1 = X @ dL_dz1.T
        return dL_dW1, dL_dz1, dL_dW2, dL_dz2

    def update_weights(self, dW1, db1, dW2, db2, learning_rate):
        """
        dW1, dW2, db1, db2 = gradients of weights and biases calculated in backwards_propagation
        learning_rate = the constant used in calculating the changes to the weights and biases
        """
        self.W1 -= dW1 * learning_rate
        self.W2 -= dW2 * learning_rate
        self.b1 -= db1 * learning_rate
        self.b2 -= db2 * learning_rate


training_data = keras.datasets.mnist.load_data() # Load data

nn = NeuralNetwork()

for image, label in training_data:
    y_one_hot = np.zeros((10, 1))
    y_one_hot[label] = 1
    a1, a2 = nn.forward_propogation(y_one_hot)
    loss = nn.calculate_loss(y_one_hot, a2)
    dW1, db1, dW2, db2 = nn.backwards_propagation(image, y_one_hot, a1, a2)
    nn.update_weights(dW1, db1, dW2, db2, learning_rate=0.01)
