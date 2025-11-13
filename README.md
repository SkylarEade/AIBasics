# Machine Learning Handwritten Number Analyzer

**Starter project for understanding basic machine learning structure and algorithms made completely from scratch using only numpy**

### What it Does
Trains and saves a neural network that can detect what number is depicted in a handwritten image

### Lessons learned
- Fundamental concepts like weights and biases
- Forward propagation (matrix multiplication, transposition and activation functions like sigmoid and relu)
- Backward propagation (derivatives and chain rule)
- Weight updating using the derivative calculations
- Gradient building using Mini-Batch gradients

### How to run
```
git clone https://github.com/SkylarEade/AIBasics.git
cd AIBasics
pip install -r requirements.txt
```
To train run:
```
python train.py
```
To run test on unseen data run:
```
python test.py
# OR TO VISUALIZE THIS TEST (on ten random samples):
python test.py -visualize
```
![Visual of Predictions](.assets/image.png)
