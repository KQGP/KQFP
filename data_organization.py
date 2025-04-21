import os
from collections import defaultdict

def organize_dataset(dataset_path):
    """Organize the dataset into a dictionary."""
    finger_dict = defaultdict(list)
    
    for file_name in os.listdir(dataset_path):
        if file_name.endswith('.tif'):
            finger_id = file_name.split('_')[0]
            finger_dict[finger_id].append(os.path.join(dataset_path, file_name))
    
    return finger_dict

if __name__ == "__main__":
    dataset_path = 'Dataset'
    finger_dict = organize_dataset(dataset_path)
    print("Dataset organized successfully!")