# KQFP — Fingerprint Matching (Siamese Network)

A Python project that trains and uses a **Siamese neural network** to compare and match fingerprints using the **FVC2004** dataset.

## ✨ Features

- 🗂️ Dataset organization by finger id
- 🔗 Positive/negative pair generation
- 🧠 CNN feature extractor + Siamese model with custom `AbsoluteDifference` layer
- 🏋️ Training pipeline
- 🔎 Inference (`compare_fingerprints`)
- 📈 Evaluation reporting accuracy

## 🧱 Tech Stack

- Python, TensorFlow/Keras
- NumPy, scikit-learn, Pillow

## 📁 Project Structure

```
data_organization.py     # Organize dataset into finger groups
data_preprocessing.py    # Create pairs & prepare data
model_training.py        # Build & train Siamese model
model_inference.py       # Inference helpers
model_evaluation.py      # Evaluate saved model
KQFP.py                  # Compare two fingerprints (entry helper)
utils.py                 # Image loading/preprocessing
fingerprint_matching_model.h5  # Trained model
Dataset/                 # FVC2004 images (.tif)
DB1_B.zip ... DB4_B.zip  # Dataset archives
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow 2.x

### Install
```bash
pip install -r requirements.txt
```

### Pipeline
Run in order:
```bash
python data_organization.py
python data_preprocessing.py
python model_training.py
python model_evaluation.py
```

### Inference
```python
from KQFP import compare_fingerprints
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.layers import Layer
import tensorflow as tf

class AbsoluteDifference(Layer):
    def call(self, inputs):
        return tf.abs(inputs[0] - inputs[1])

with custom_object_scope({'AbsoluteDifference': AbsoluteDifference}):
    model = load_model('fingerprint_matching_model.h5')

print(compare_fingerprints('a.tif', 'b.tif', model))
```
