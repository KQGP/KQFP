import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.layers import Layer
import tensorflow as tf
from utils import load_and_preprocess_pair

# Define the custom layer
class AbsoluteDifference(Layer):
    def __init__(self, **kwargs):
        super(AbsoluteDifference, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.abs(inputs[0] - inputs[1])

def compare_fingerprints(image1_path, image2_path, model, threshold=0.5):
    """
    Compare two fingerprint images and determine if they match.
    
    Args:
        image1_path (str): Path to the first fingerprint image.
        image2_path (str): Path to the second fingerprint image.
        model: Trained Siamese Network model.
        threshold (float): Similarity threshold to determine a match (default: 0.5).
    
    Returns:
        str: "Match" or "No Match" based on the similarity score.
        float: Similarity score between the two fingerprints.
    """
    # Load and preprocess the pair of images
    image1, image2 = load_and_preprocess_pair(image1_path, image2_path)

    # Reshape images to match the model's input shape
    image1 = np.expand_dims(image1, axis=0)  # Add batch dimension
    image2 = np.expand_dims(image2, axis=0)  # Add batch dimension

    # Compute similarity score using the model
    similarity_score = model.predict([image1, image2])[0][0]

    # Determine if the fingerprints match
    if similarity_score >= threshold:
        return "Match", similarity_score
    else:
        return "No Match", similarity_score

if __name__ == "__main__":
    # Load the trained model with the custom layer
    with custom_object_scope({'AbsoluteDifference': AbsoluteDifference}):
        model = load_model('fingerprint_matching_model.h5')

    # Paths to the two fingerprint images
    image1_path = 'Pic.tif'  # Replace with the path to your first fingerprint image
    image2_path = 'Pic2.tif'  # Replace with the path to your second fingerprint image

    # Compare the fingerprints
    result, score = compare_fingerprints(image1_path, image2_path, model)
    print(f"Result: {result}")
    print(f"Similarity Score: {score:.4f}")