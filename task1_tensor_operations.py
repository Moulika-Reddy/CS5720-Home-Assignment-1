import tensorflow as tf

# 1. Create a random tensor with shape (4, 6)
tensor = tf.random.uniform(shape=(4, 6))

print("Original Tensor:")
print(tensor)


# 2. Find the rank and shape of the original tensor
original_rank = tf.rank(tensor)
original_shape = tf.shape(tensor)

print("\nOriginal Tensor Rank:", original_rank.numpy())
print("Original Tensor Shape:", original_shape.numpy())


# 3. Reshape the tensor from (4, 6) to (2, 3, 4)
reshaped_tensor = tf.reshape(tensor, shape=(2, 3, 4))

print("\nReshaped Tensor:")
print(reshaped_tensor)

print("Reshaped Tensor Rank:", tf.rank(reshaped_tensor).numpy())
print("Reshaped Tensor Shape:", tf.shape(reshaped_tensor).numpy())


# Transpose the tensor from shape (2, 3, 4) to (3, 2, 4)
# perm=[1, 0, 2] changes the order of the first two dimensions
transposed_tensor = tf.transpose(
    reshaped_tensor,
    perm=[1, 0, 2]
)

print("\nTransposed Tensor:")
print(transposed_tensor)

print("Transposed Tensor Rank:", tf.rank(transposed_tensor).numpy())
print("Transposed Tensor Shape:", tf.shape(transposed_tensor).numpy())


# 4. Create a smaller tensor with shape (1, 4)
small_tensor = tf.random.uniform(shape=(1, 4))

print("\nSmall Tensor:")
print(small_tensor)

print("Small Tensor Shape:", tf.shape(small_tensor).numpy())


# Add the smaller tensor to the larger tensor.
# TensorFlow automatically broadcasts the smaller tensor
# to match the compatible dimensions of the larger tensor.
result = transposed_tensor + small_tensor

print("\nResult After Broadcasting and Addition:")
print(result)

print("Result Shape:", tf.shape(result).numpy())