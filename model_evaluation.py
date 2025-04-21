import numpy as np
import tensorflow as tf  # Import TensorFlow
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.layers import Layer  # Import the Layer class
from data_preprocessing import prepare_data, create_pairs
from data_organization import organize_dataset

# Define the custom layer
class AbsoluteDifference(Layer):
    def __init__(self, **kwargs):
        super(AbsoluteDifference, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.abs(inputs[0] - inputs[1])  # Use tf.abs

if __name__ == "__main__":
    dataset_path = 'Dataset'
    finger_dict = organize_dataset(dataset_path)
    pairs = create_pairs(finger_dict)
    data, labels = prepare_data(pairs)

    # Load the model with the custom layer
    with custom_object_scope({'AbsoluteDifference': AbsoluteDifference}):
        model = load_model('fingerprint_matching_model.h5')

    test_loss, test_accuracy = model.evaluate([data[0], data[1]], labels)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")