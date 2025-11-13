import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.W1 = np.random.randn(784,512) * 0.1 # Weight initialziation
        self.W2 = np.random.randn(512,128) * 0.1
        self.W3 = np.random.randn(128, 10) * 0.1
        self.b1 = np.zeros((512, 1)) # Bias initialization
        self.b2 = np.zeros((128, 1))
        self.b3 = np.zeros((10, 1))

    def sigmoid(self, arr):
        return 1/ (1+np.exp(-arr)) # Sigmoid function for compressing the weights into 0 - 1 range
    
    def relu(self, z):
        return np.maximum(0, z)

    def forward_propagation(self, X):
        """
        X = 784 x 1 array representing the 28 x 28 pixel image
        W = Weight
        b = Bias
        """
        z1 = self.W1.T @ X + self.b1
        a1 = self.relu(z1)
        z2 = self.W2.T @ a1 + self.b2
        a2 = self.relu(z2)
        z3 = self.W3.T @ a2 + self.b3
        a3 = self.sigmoid(z3)
        return a1, a2, z1, a3, z2

    def calculate_loss(self, y_one_hot, prediction):
        """
        y_one_hot = an array of n zeros except the correct value inside of the image with a value of one where n is the number of possible predictions
        prediction = array of predictions from the model
        """
        return np.mean((prediction - y_one_hot) ** 2)
        
    def relu_derivative(self, z):
        """
        Relu derivative is simply a 1 or 0 depending on if z is greater than 0. dz = 1 if z > 0 else 0
        """
        return (z > 0).astype(float)

    def backwards_propagation(self, X, y_one_hot, a1, a2, a3, z1, z2):
        """
        Backwards propagation will determine how to tweak the weights and biases so the model can learn from its mistakes
        a1, a2 = the activations of the neuron layers
        X = 784 x 1 array representing the 28 x 28 pixel image
        y_one_hot = actual value presented in the image in the form of an array where the number's index has the value 1
        z1 = included for derivative calculation for relu

        Derivative of our MSE Loss function (L = (1/n) * summation(a2 - y)^2) = (2/n) x (a2 - y)
        We will simply absorb the 2/n for now
        So the new derivative becomes a2 - y

        The chain rule will be used for the rest of the derivates, so in this case dL/dz2 = dL/da2 x da2/dz2
        Calculating da2/dz2 we will use the sigmoid derivative property which states if a = sigmoid(z) then:
        derivative = a * (1-a)
        Since we have dL/da2 already calculate (see above) we will write this out as dL_da2 * a2 * (1 - a2)

        For dL/DW2 we know that z2 = W2.T @ a1 + b2, using the chain rule we can write the derivate as:
        dL/dW2 = dL/dz2 * dz2/dW2
        the derivative of z2 with respect to W2 will simply be a1 and we have already calculated dL_dz2 so it will be written:
        dl_DW2 = a1 @ dl_dz2.T

        Finally, dL_dz = dL_db because b is a constant and when derived will become equal to 1.
        """
        # Calculate derivate of weight 3
        dL_da3 = a3 - y_one_hot
        dL_dz3 = dL_da3 * a3 * (1 - a3) # doubles as dL_db2 because when derived b becomes 1
        dL_dW3 = a2 @ dL_dz3.T

        # Calculate derivative of weight 2
        dL_da2 = self.W3 @ dL_dz3
        dL_dz2 = dL_da2 * self.relu_derivative(z2) # doubles as dL_db1 because when derived b becomes 1
        dL_dW2 = a1 @ dL_dz2.T

        # Calculate derivative of weight 1
        dL_da1 = self.W2 @ dL_dz2
        dL_dz1 = dL_da1 * self.relu_derivative(z1) # doubles as dL_db1 because when derived b becomes 1
        dL_dW1 = X @ dL_dz1.T
        return dL_dW1, dL_dz1, dL_dW2, dL_dz2, dL_dW3, dL_dz3

    def update_weights(self, dW1, db1, dW2, db2, dW3, db3, learning_rate):
        """
        dW1, dW2, db1, db2 = gradients of weights and biases calculated in backwards_propagation
        learning_rate = the constant used in calculating the changes to the weights and biases
        """
        self.W1 -= dW1 * learning_rate
        self.W2 -= dW2 * learning_rate
        self.W3 -= dW3 * learning_rate
        self.b1 -= db1 * learning_rate
        self.b2 -= db2 * learning_rate
        self.b3 -= db3 * learning_rate

    def push_update_weight(self, dW1_sum, dW2_sum, dW3_sum, db1_sum, db2_sum, db3_sum, size, learning_rate):
        dW1_avg = dW1_sum / size
        dW2_avg = dW2_sum / size
        dW3_avg = dW3_sum / size
        db1_avg = db1_sum / size
        db2_avg = db2_sum / size
        db3_avg = db3_sum / size
        
        self.update_weights(dW1_avg, db1_avg, dW2_avg, db2_avg, dW3_avg, db3_avg, learning_rate)
        return (np.zeros_like(dW1_sum), np.zeros_like(dW2_sum), np.zeros_like(dW3_sum), np.zeros_like(db1_sum), np.zeros_like(db2_sum),np.zeros_like(db3_sum))
    
    def train(self, X_train, y_train, epochs, learning_rate=0.01, batch_size=32):
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_correct = 0
            loss_sum = 0
            correct_count = 0
            dW1_sum = 0
            dW2_sum = 0
            dW3_sum = 0
            db1_sum = 0
            db2_sum = 0
            db3_sum = 0
            for j in range(len(X_train)):
                image = X_train[j]
                label = y_train[j]
                img = image.reshape(784, 1) / 255
                y_one_hot = np.zeros((10, 1))
                y_one_hot[label] = 1
                a1, a2, z1, a3, z2 = self.forward_propagation(img)
                guess = np.argmax(a3)
                correct_count += (1 if guess == label else 0)
                loss_sum += self.calculate_loss(y_one_hot, a3)
                dW1, db1, dW2, db2, dW3, db3 = self.backwards_propagation(img, y_one_hot, a1, a2, a3, z1, z2)
                dW1_sum += dW1
                dW2_sum += dW2
                dW3_sum += dW3
                db1_sum += db1
                db2_sum += db2
                db3_sum += db3
                if (j+1) % batch_size == 0: # Every 32 predictions change batch and report total loss
                    epoch_loss += loss_sum
                    epoch_correct += correct_count
                    loss_sum = 0
                    dW1_sum, dW2_sum, dW3_sum, db1_sum, db2_sum, db3_sum = self.push_update_weight(dW1_sum, dW2_sum, dW3_sum, db1_sum, db2_sum, db3_sum, batch_size, learning_rate)
                    correct_count = 0
            if (len(X_train) % batch_size) != 0: # Get any remnant changes that didnt fit into full final batch
                size = (j % batch_size) + 1
                dW1_sum, dW2_sum, dW3_sum, db1_sum, db2_sum, db3_sum = self.push_update_weight(dW1_sum, dW2_sum, dW3_sum, db1_sum, db2_sum, db3_sum, size, learning_rate)

            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{epochs} completed:\nAverage Loss: {(epoch_loss/len(X_train)):.4f}\nAccuracy: {(epoch_correct/len(X_train) * 100):.2f}%")
            print(f"{'='*60}\n")

    def save_model(self, filepath="models/mnist_model.npz"):
        np.savez(filepath, W1=self.W1, W2=self.W2, W3=self.W3, b1=self.b1, b2=self.b2, b3=self.b3)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath="models/mnist_model.npz"):
        data = np.load(filepath)
        self.W1 = data["W1"]
        self.W2 = data["W2"]
        self.W3 = data["W3"]
        self.b1 = data["b1"]
        self.b2 = data["b2"]
        self.b2 = data["b3"]
