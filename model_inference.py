import os
import random
import numpy as np
import tensorflow as tf
from collections import defaultdict
from PIL import Image

# Organize the dataset
def organize_dataset(dataset_path):
    """Organize the dataset into a dictionary."""
    finger_dict = defaultdict(list)
    
    for file_name in os.listdir(dataset_path):
        if file_name.endswith('.tif'):
            finger_id = file_name.split('_')[0]
            finger_dict[finger_id].append(os.path.join(dataset_path, file_name))
    
    return finger_dict

# Load and preprocess images
def load_and_preprocess_image(image_path, target_size=(128, 128)):
    """Load and preprocess a .tif image."""
    # Load the image using Pillow
    image = Image.open(image_path)
    
    # Convert to grayscale if necessary
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize the image
    image = image.resize(target_size)
    
    # Convert to numpy array
    image = np.array(image)
    
    # Normalize to [0, 1]
    image = image / 255.0
    
    # Add a channel dimension (required for TensorFlow)
    image = np.expand_dims(image, axis=-1)
    
    return image

# Create pairs
def create_pairs(finger_dict):
    """Create positive and negative pairs."""
    positive_pairs = []
    negative_pairs = []

    finger_ids = list(finger_dict.keys())

    for finger_id, file_paths in finger_dict.items():
        # Positive pairs
        for i in range(len(file_paths)):
            for j in range(i + 1, len(file_paths)):
                positive_pairs.append((file_paths[i], file_paths[j], 1))
        
        # Negative pairs
        for file_path in file_paths:
            other_finger_id = random.choice([f for f in finger_ids if f != finger_id])
            other_file_path = random.choice(finger_dict[other_finger_id])
            negative_pairs.append((file_path, other_file_path, 0))

    pairs = positive_pairs + negative_pairs
    random.shuffle(pairs)
    return pairs

# Prepare data
def prepare_data(pairs):
    """Prepare data for training."""
    images_a = []
    images_b = []
    labels = []
    
    for pair in pairs:
        images_a.append(load_and_preprocess_image(pair[0]))
        images_b.append(load_and_preprocess_image(pair[1]))
        labels.append(pair[2])
    
    return [np.array(images_a), np.array(images_b)], np.array(labels)

# Main script
if __name__ == "__main__":
    dataset_path = 'Dataset'
    finger_dict = organize_dataset(dataset_path)
    pairs = create_pairs(finger_dict)
    data, labels = prepare_data(pairs)
    print("Data preparation completed successfully!")