import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("Available devices:")
print(tf.config.list_physical_devices())


# Create a constant tensor
tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])

# Perform a simple operation
result = tensor + tensor

print("Result:")
print(result)