import tensorflow as tf
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# CS5720 - Neural Network & Deep Learning
# Home Assignment 1
# Part II - Task 3: MNIST with Adam vs SGD
# Student Name: Moulika Reddy
# ---------------------------------------------------------

# 1. Load the MNIST handwritten digit dataset.
# x_train and x_test contain the digit images.
# y_train and y_test contain the corresponding digit labels.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Normalize pixel values from the range 0-255 to 0-1.
# This helps the neural network train more efficiently.
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. Create a function that builds the same neural network architecture.
# We use the same model structure for both optimizers so that
# the optimizer is the main difference being compared.
def create_model():
    model = tf.keras.Sequential([
        # Flatten converts each 28x28 image into a 1D vector of 784 values.
        tf.keras.layers.Flatten(input_shape=(28, 28)),

        # Hidden layer with 128 neurons and ReLU activation.
        tf.keras.layers.Dense(128, activation="relu"),

        # Output layer with 10 neurons, one for each digit from 0 to 9.
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    return model


# ---------------------------------------------------------
# MODEL 1: SGD OPTIMIZER
# ---------------------------------------------------------

# Create the first model.
sgd_model = create_model()

# Compile the model using SGD.
# Sparse categorical cross-entropy is used because
# the MNIST labels are integer values such as 0, 1, 2, ..., 9.
sgd_model.compile(
    optimizer=tf.keras.optimizers.SGD(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nTraining Model 1 using SGD...\n")

# Train the SGD model for 5 epochs.
# validation_split=0.2 reserves 20% of the training data
# for validation during training.
sgd_history = sgd_model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_split=0.2,
    batch_size=32,
    verbose=1
)


# ---------------------------------------------------------
# MODEL 2: ADAM OPTIMIZER
# ---------------------------------------------------------

# Create a new model with the same architecture.
adam_model = create_model()

# Compile the model using Adam.
adam_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nTraining Model 2 using Adam...\n")

# Train the Adam model using the same settings.
adam_history = adam_model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_split=0.2,
    batch_size=32,
    verbose=1
)


# ---------------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------------

# Evaluate both trained models on the test dataset.
sgd_test_loss, sgd_test_accuracy = sgd_model.evaluate(
    x_test,
    y_test,
    verbose=0
)

adam_test_loss, adam_test_accuracy = adam_model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("\nTest Results:")
print("SGD Test Accuracy:", sgd_test_accuracy)
print("Adam Test Accuracy:", adam_test_accuracy)


# ---------------------------------------------------------
# ACCURACY COMPARISON PLOT
# ---------------------------------------------------------

# Epoch numbers for the x-axis.
epochs = range(1, 6)

plt.figure(figsize=(10, 6))

# SGD accuracy curves.
plt.plot(
    epochs,
    sgd_history.history["accuracy"],
    marker="o",
    label="SGD Training Accuracy"
)

plt.plot(
    epochs,
    sgd_history.history["val_accuracy"],
    marker="o",
    label="SGD Validation Accuracy"
)

# Adam accuracy curves.
plt.plot(
    epochs,
    adam_history.history["accuracy"],
    marker="o",
    label="Adam Training Accuracy"
)

plt.plot(
    epochs,
    adam_history.history["val_accuracy"],
    marker="o",
    label="Adam Validation Accuracy"
)

plt.title("MNIST Accuracy Comparison: SGD vs Adam")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()

# Save the graph for the GitHub repository and README.
plt.savefig("optimizer_comparison.png")

# Display the graph.
plt.show()