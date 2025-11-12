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

    def calculate_loss(self, label, prediction):
        """
        label = the correct value inside of the image
        prediction = array of predictions from the model
        """
        one_hot = np.zeros((10, 1))
        one_hot[label] = 1
        loss = np.mean((prediction - one_hot) ** 2)
        return loss


        


(train_images, train_labels), _ = keras.datasets.mnist.load_data() # Load data

nn = NeuralNetwork()