import tensorflow as tf
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# CS5720 - Neural Network & Deep Learning
# Home Assignment 1
# Part II - Task 2: Loss Functions
# Student Name: Moulika Reddy
# ---------------------------------------------------------

# 1. Define the true class labels.
# Each row represents one example with three possible classes.
# [1, 0, 0] means Class 1 is the correct class.
# [0, 1, 0] means Class 2 is the correct class.
# [0, 0, 1] means Class 3 is the correct class.
y_true = tf.constant([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
])

# 2. Define the first set of predictions.
# These predictions are relatively close to the correct answers.
y_pred_1 = tf.constant([
    [0.9, 0.05, 0.05],
    [0.1, 0.8, 0.1],
    [0.05, 0.05, 0.9]
])

# 3. Define a second set of predictions.
# These predictions are less accurate than the first set.
y_pred_2 = tf.constant([
    [0.6, 0.2, 0.2],
    [0.2, 0.5, 0.3],
    [0.2, 0.3, 0.5]
])

# Create the loss-function objects.
mse = tf.keras.losses.MeanSquaredError()
cce = tf.keras.losses.CategoricalCrossentropy()

# 4. Calculate MSE for both prediction sets.
mse_loss_1 = mse(y_true, y_pred_1)
mse_loss_2 = mse(y_true, y_pred_2)

# 5. Calculate Categorical Cross-Entropy for both prediction sets.
cce_loss_1 = cce(y_true, y_pred_1)
cce_loss_2 = cce(y_true, y_pred_2)

# 6. Print the calculated loss values.
print("Prediction Set 1:")
print("MSE Loss:", mse_loss_1.numpy())
print("Categorical Cross-Entropy Loss:", cce_loss_1.numpy())

print("\nPrediction Set 2:")
print("MSE Loss:", mse_loss_2.numpy())
print("Categorical Cross-Entropy Loss:", cce_loss_2.numpy())

# 7. Prepare data for the bar chart.
labels = [
    "MSE - Prediction 1",
    "CCE - Prediction 1",
    "MSE - Prediction 2",
    "CCE - Prediction 2"
]

loss_values = [
    mse_loss_1.numpy(),
    cce_loss_1.numpy(),
    mse_loss_2.numpy(),
    cce_loss_2.numpy()
]

# 8. Plot the loss values using Matplotlib.
plt.figure(figsize=(9, 5))
plt.bar(labels, loss_values)

plt.title("Comparison of MSE and Categorical Cross-Entropy Loss")
plt.xlabel("Loss Function and Prediction Set")
plt.ylabel("Loss Value")
plt.xticks(rotation=20)

plt.tight_layout()

# Save the graph so it can be included in the GitHub repository.
plt.savefig("loss_comparison.png")

# Display the graph.
plt.show()