import main
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

parser = argparse.ArgumentParser()
parser.add_argument("-visualize", action="store_true", help="Visualize the test set")
args = parser.parse_args()
(x_train, y_train),(x_test, y_test) = keras.datasets.mnist.load_data()
nn= main.NeuralNetwork()
nn.load_model("models/test.npz")
correct = 0

if args.visualize:
    print("Generating visualizations...")
    
    fig = plt.figure(figsize=(15, 6))
    fig.patch.set_facecolor('#2e3440')
    
    axes = fig.subplots(2, 5)
    
    for i, ax in enumerate(axes.flat):
        idx = np.random.randint(0, len(x_test))
        image = x_test[idx]
        label = y_test[idx]
        
        img = image.reshape(784, 1) / 255
        _, _, _, a3, _ = nn.forward_propagation(img)
        guess = np.argmax(a3)
        confidence = a3[guess, 0] * 100
        
        ax.imshow(image, cmap='binary', interpolation='nearest')
        
        is_correct = (guess == label)
        color = '#a3be8c' if is_correct else '#bf616a' 
        symbol = '✓' if is_correct else '✗'
        
        ax.set_title(
            f"{symbol} Prediction: {guess} ({confidence:.1f}%)\n"
            f"Actual: {label}",
            color=color,
            fontsize=11,
            fontweight='bold',
            pad=8
        )
        ax.set_xticks([])
        ax.set_yticks([])
        
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)
    plt.suptitle(
        'Neural Network Predictions',
        fontsize=18,
        fontweight='bold',
        color='white',
        y=0.98
    )
 
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
else: 
    for i in range(len(x_test)):
        image = x_test[i]
        y_one_hot = np.zeros((10,1))
        y_one_hot[y_test[i]] = 1
        img = image.reshape(784, 1) / 255
        _, _, _, a3, _ = nn.forward_propagation(img)
        guess = np.argmax(a3)
        if guess == y_test[i]:
            correct += 1
    print(f"\n{'='*60}")
    print(f"Total Correct Guesses: {correct} out of {len(x_test)}\nAccuracy: {(correct/len(x_test) * 100):.2f}%")
    print(f"{'='*60}\n")
