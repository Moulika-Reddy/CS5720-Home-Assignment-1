import tensorflow as tf
from datetime import datetime
import os

# ---------------------------------------------------------
# CS5720 - Neural Network & Deep Learning
# Home Assignment 1
# Part II - Task 4: MNIST with TensorBoard
# Student Name: Moulika Reddy
# ---------------------------------------------------------

# 1. Load the MNIST handwritten digit dataset.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Normalize pixel values from 0-255 to 0-1.
# This helps the neural network train more efficiently.
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. Create a simple neural network.
model = tf.keras.Sequential([
    # Convert each 28x28 image into a vector of 784 values.
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),

    # Hidden layer with 128 neurons and ReLU activation.
    tf.keras.layers.Dense(128, activation="relu"),

    # Output layer with 10 neurons for digits 0 through 9.
    tf.keras.layers.Dense(10, activation="softmax")
])

# 4. Compile the model.
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------------------------------------------------------
# TENSORBOARD LOGGING
# ---------------------------------------------------------

# Create a unique folder for this training run.
# The logs will be stored inside logs/fit/.
log_dir = os.path.join(
    "logs",
    "fit",
    datetime.now().strftime("%Y%m%d-%H%M%S")
)

# Create the TensorBoard callback.
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
)

print("TensorBoard logs will be stored in:")
print(log_dir)

# ---------------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------------

# Train the model for 5 epochs.
# validation_split=0.2 uses 20% of the training data for validation.
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    callbacks=[tensorboard_callback],
    verbose=1
)

# ---------------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------------

# Evaluate the trained model on the test dataset.
test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("\nTest Results:")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

print("\nTraining complete.")
print("Run TensorBoard using:")
print("tensorboard --logdir logs/fit")