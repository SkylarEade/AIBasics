import main
import numpy as np
from tensorflow import keras


(x_train, y_train),(x_test, y_test) = keras.datasets.mnist.load_data()
nn= main.NeuralNetwork()
nn.load_model()
correct = 0
for i in range(len(x_test)):
    image = x_test[i]
    y_one_hot = np.zeros((10,1))
    y_one_hot[y_test[i]] = 1
    img = image.reshape(784, 1) / 255
    a1, a2, z1 = nn.forward_propagation(img)
    guess = np.argmax(a2)
    if guess == y_test[i]:
        correct += 1

print(f"\n{'='*60}")
print(f"Total Correct Guesses: {correct} out of {len(x_test)}\nAccuracy: {(correct/len(x_test) * 100):.2f}%")
print(f"{'='*60}\n")
