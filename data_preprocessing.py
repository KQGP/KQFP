# data_preprocessing.py
import random
import numpy as np
from utils import load_and_preprocess_image

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

if __name__ == "__main__":
    from data_organization import organize_dataset
    dataset_path = 'Dataset'
    finger_dict = organize_dataset(dataset_path)
    pairs = create_pairs(finger_dict)
    data, labels = prepare_data(pairs)
    print(f"Prepared data: {len(data[0])} samples.")