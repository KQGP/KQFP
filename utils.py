import numpy as np
from PIL import Image

def load_and_preprocess_image(image_path, target_size=(128, 128)):
    """
    Load and preprocess a single image.
    
    Args:
        image_path (str): Path to the image file.
        target_size (tuple): Target size for resizing the image (default: (128, 128)).
    
    Returns:
        np.array: Preprocessed image as a numpy array.
    """
    # Use Pillow to open the .tif image
    image = Image.open(image_path)
    
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize the image
    image = image.resize(target_size)
    
    # Convert to numpy array and normalize to [0, 1]
    image = np.array(image) / 255.0
    
    # Add a channel dimension (required for TensorFlow)
    image = np.expand_dims(image, axis=-1)
    
    return image

def load_and_preprocess_pair(image1_path, image2_path, target_size=(128, 128)):
    """
    Load and preprocess a pair of images.
    
    Args:
        image1_path (str): Path to the first image file.
        image2_path (str): Path to the second image file.
        target_size (tuple): Target size for resizing the images (default: (128, 128)).
    
    Returns:
        tuple: A tuple containing the two preprocessed images as numpy arrays.
    """
    # Preprocess the first image
    image1 = load_and_preprocess_image(image1_path, target_size)
    
    # Preprocess the second image
    image2 = load_and_preprocess_image(image2_path, target_size)
    
    return image1, image2