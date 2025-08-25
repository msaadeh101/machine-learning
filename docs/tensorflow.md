# TensorFlow Guide

This guide walks you from **zero knowledge of TensorFlow** to confidently building and training neural networks. Covers the foundations, build practical models, and explain concepts along the way.


## Table of Contents

1. [What is TensorFlow?](#what-is-tensorflow)
1. [What is a Tensor?](#what-is-a-tensor)
1. [Installing TensorFlow](#installing-tensorflow)
1. [Tensor Operations](#tensor-operations)
1. [Building Your First Neural Network](#building-your-first-neural-network)
1. [Understanding the Keras API](#understanding-the-keras-api)
1. [Model Evaluation & Prediction](#model-evaluation--prediction)
1. [Callbacks & TensorBoard](#callbacks--tensorboard)
1. [Saving and Loading Models](#saving-and-loading-models)


## What is TensorFlow?

**TensorFlow** is an open-source library developed by Google for building and training machine learning models, especially neural networks. It provides tools for:

- Creating computational graphs
- Running operations on GPUs/TPUs
- High-level APIs for fast model building (`tf.keras`) - based on the original Keras API for deep learning.

### Keras

Take this Model Architecture, we will apply both a functional API approach and a Sequential model approach with the same data (MNST)

```txt
(input: 784-dimensional vectors)
       ↧
[Dense (64 units, relu activation)]
       ↧
[Dense (64 units, relu activation)]
       ↧
[Dense (10 units, softmax activation)]
       ↧
(output: logits of a probability distribution over 10 classes)
```
- input: typically from flattened grayscale images. 28x28=784, input layer accepts batch size of 784
- Dense 64: each fully connected layer with 64 neurons. Applies reLU activation `f(x) = max(0,x)`
- Dense 10: Final layer with 10 neurons, one for each class (0-9 handwritten digits), 
- output: shape is 10

#### Functional API

- The **Keras functional API** is the way to create models that are more flexible than keras.Sequential. Can handle non-linear topology, shared layers, and multiple inputs/outputs.

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Input layer
inputs = Input(shape=(784,))

# Hidden layers
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)

# Output layer
outputs = Dense(10, activation='softmax')(x)

# Define the model
model = Model(inputs=inputs, outputs=outputs)

model.summary()
```

#### Sequential

- Use Sequential when you have a simple stack.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(64, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.summary()

```

## What is a Tensor?

A **tensor** is the primary data structure. It is a multi-dimensional array similar to NumPy arrays but with extra capabilities (like GPU acceleration). 

- There are three core properties: 
    - `Rank (ndim)`: Number of dimensions the tensor has.
        - `scalar` = 0
        - `vector` = 1
        - `matrix` = 2
    - `Shape`: Tuple of integers telling you the size of the tensor along each dimension.
        - `()` = a scalar shape/empty tuple.
        - `(5,)` = a vector with 5 elements.
        - `(3, 2)` = a matrix with shape of 3x2.
    - `Data type (dtype)`
        - Can be `float32`, `int32`, `string`.


| Tensor Rank | Example            | Shape        |  Description   |
|-------------|--------------------|--------------| ---------------|
| 0 (scalar)  | `42`               | `()`         |  Single number that has no direction. zero dimensions.  |
| 1 (vector)  | `[1.0, 2.0, 3.0]`  | `(3,)`       |  One dimensional tensor, ordered list of numbers. 1 dimensional  |
| 2 (matrix)  | `[[1, 2], [3, 4]]` | `(2, 2)`     | Rows and columns, two dimensional    |
| 3+          | Image data, etc.   | `(batch, height, width, channels)` | Used for things like image data  |

```python
import tensorflow as tf

scalar = tf.constant(42)
vector = tf.constant([1.0, 2.0, 3.0])
matrix = tf.constant([[1, 2], [3, 4]])
tensor = tf.constant([[[1], [2]], [[3], [4]]])  # Shape (2, 2, 1)

print("Scalar:", scalar)
print("Vector:", vector)
print("Matrix shape:", matrix.shape)
```

- Tensors provide unified data representation for consisent and seamless handling of data.
- Tensors are designed to be efficiently stored on GPUs for parallel processing.
- Tensors are integral to how Neural networks learn.


## Installing Tensorflow

```bash
pip install tensorflow
```

## Tensorflow operations

- Manipulate tensors like a NumPy object

- `tf.add` is the same as `(a + b)`.
- `tf.multiply` is element-wise multiplication `(a * b)`.
- `tf.matmul` is matrix manipulation `(a @ b)`. Number of rows in matrix must match columns of second. **Computes the weighted sum of the inputs for each neuron in the layer**.
    - The rows of the first matrix often represent a batch of input data.
    - The second matrix represents the learnable weights of the neural network layer.

```python
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])

print("Add:\n", tf.add(a, b))
print("Multiply:\n", tf.multiply(a, b))
print("Matrix Multiply:\n", tf.matmul(a, b))

```

- Tensorflow methods like `tf.pow`, `tf.reduce_mean`, `tf.reduce_sum` helps you with **feature engineering**: Coming up with new actionalable features to train the model with.

```python
cart_totals = tf.constant([120.0, 80.0, 150.0, 60.0])
avg_cart_value = tf.reduce_mean(cart_totals)
```

- `tf.transpose()` switches the rows and columns. It reorders the dimensions of a tensor.
    - For example, matrix multiplcation requires inner dimensions to match, so you may need to transpose them.
    - Certain orders/shape are required for `keras.layers.LTSM` or a `Transformer`.

## Building Your First Neural Network

- Use the MNIST handwritten digit dataset.

```python
from tensorflow.keras.datasets import mnist
# minst contains images of handwritten digits
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize by dividing by 255, to get a number between 0 and 1
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define the model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # 10 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model: Takes the untrained data, adjusts internal weights and biases over multiple epochs
model.fit(x_train, y_train, epochs=5, validation_split=0.1)

```

- Using that model to make predictions:

```python
import numpy as np

# Assuming your model has already been trained with model.fit()

# Get a single image from the test set. Let's take the first one.
# It's important to keep the dimensions that the model expects.
# x_test[0] has shape (28, 28). The model expects a batch, so we reshape it to (1, 28, 28).
img = x_test[0]
img = np.expand_dims(img, axis=0) # adds a dimension to make it a batch of 1

# Make a prediction
predictions = model.predict(img)

# The output is a list of 10 probabilities (from the softmax layer)
print("Prediction probabilities:", predictions)

# To get the final predicted digit, we find the index of the highest probability
predicted_digit = np.argmax(predictions)
print("Predicted digit:", predicted_digit)

# You can also check the true label to see if the prediction was correct
true_digit = np.argmax(y_test[0])
print("True digit:", true_digit)
```

## Understanding the Keras API

- `tf.keras` is the high level API that simplifies model building. Pre-built, user-friendly tools for training deep learning models.

### Sequential
- Sequential model: Simple stack of layers.
    - You create an empty `Sequential` then `.add()` layers one by one.
    - The output of one layer is automatically used as input for the next.
    - Used for basic classifier, single input/output.

```python
import tensorflow as tf

# Create a Sequential model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])
```
- `tf.keras.layers.Dense`: A fully connected layer.
- `activation='relu'`: A common non-linear activation function.
- `input_shape`: You only need to specify this for the very first layer.

### Functional API
- Funcitonal API: For complex models with branches.
    - More flexible for complex architectures. Layers are functions.
    - Good for models with multiple inputs.

```python
import tensorflow as tf

# Define the input layer
input_tensor = tf.keras.Input(shape=(784,))

# A shared layer (e.g., a common feature extractor)
shared_layer = tf.keras.layers.Dense(64, activation='relu')(input_tensor)

# One branch for output A
output_a = tf.keras.layers.Dense(10, activation='softmax')(shared_layer)

# One branch for output B
output_b = tf.keras.layers.Dense(5, activation='softmax')(shared_layer)

# Combine the inputs and outputs to create the model
model = tf.keras.Model(inputs=input_tensor, outputs=[output_a, output_b])
```

### Compile    
- Model.compile(): Sets optimizer, loss, metrics.
    - You need to compile the model before you train it.
    - Configures the model's learning process.

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

### Fit
- Model.fit(): Starts training
    - Starts training loop, where x is training input data, y is labels. Inclueds epochs and batch_size

```python
# Assume X_train and y_train are your training data
model.fit(x=X_train, y=y_train, epochs=10, batch_size=32)
```

### Evaluate
- Model.evaluate(): Checks performance on test data.
    - Checks performance on a separate (test) dataset. The model should not have seen it before.

```python
# Assume X_test and y_test are your test data
loss, accuracy = model.evaluate(x=X_test, y=y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
```


## Model Evaluation & Prediction

```python
# Evaluate on test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)

# Predict a sample
import numpy as np
prediction = model.predict(np.expand_dims(x_test[0], axis=0))
print("Predicted class:", tf.argmax(prediction[0]).numpy())
```

## Callbacks & TensorBoard

- Tack training and log metrics with **TensorBoard**. TensorBoard is a web-based dashboard that lets you visualize traning metrics like loss/accuracy, monitor learning in real time, compare multiple training runs, inspect the model graphs, weights, histograms, etc.

```python
from tensorflow.keras.callbacks import TensorBoard
import datetime

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = TensorBoard(log_dir=log_dir)

model.fit(x_train, y_train, epochs=5, callbacks=[tensorboard])
```

- Start the TensorBoard:

```bash
tensorboard --logdir=logs/fit
```

## Saving and Loading Models

- Saving and Loading a model is essential for reusing the trained model, collaboration and deployment.
    - Note, `.h5` is HDF5 format, standard for storing large numerical data.
    - Included in the saved file is:
        - Architecture (layers, activations, input shape)
        - Model weights (learned parameters)
        - Optimizer state (so you can resume training)
        - compilation info (loss, metrics)

```python
# Train and save
model.fit(X_train, y_train, epochs=10)
model.save('my_model.h5')

# ... Later, or in another script
from tensorflow.keras.models import load_model
model = load_model('my_model.h5')
predictions = model.predict(X_test)
```
- Needed for serving with `TF Lite `or `TF.js`