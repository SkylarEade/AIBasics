import main
from tensorflow import keras

(x_train, y_train),(x_test, y_test) = keras.datasets.mnist.load_data()
nn = main.NeuralNetwork()
epochs = 10
learning_rate = 0.1
batch_size = 32
nn.train(x_train, y_train, epochs, learning_rate, batch_size)
filepath = "models/mnist_model.npz"
nn.save_model(filepath)