import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Layer  # Import the Layer class
from data_preprocessing import prepare_data, create_pairs
from data_organization import organize_dataset

class AbsoluteDifference(Layer):
    """Custom layer to compute the absolute difference between two tensors."""
    def __init__(self, **kwargs):
        super(AbsoluteDifference, self).__init__(**kwargs)

    def call(self, inputs):
        """Compute the absolute difference between two tensors."""
        return tf.abs(inputs[0] - inputs[1])

def build_cnn(input_shape):
    """Build a CNN model for feature extraction."""
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu')
    ])
    return model

def build_siamese_network(input_shape):
    """Build a Siamese Network."""
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)

    cnn = build_cnn(input_shape)
    encoded_a = cnn(input_a)
    encoded_b = cnn(input_b)

    # Use the custom AbsoluteDifference layer
    distance = AbsoluteDifference()([encoded_a, encoded_b])

    # Add a dense layer to classify the distance
    output = layers.Dense(1, activation='sigmoid')(distance)

    model = Model(inputs=[input_a, input_b], outputs=output)
    return model

if __name__ == "__main__":
    dataset_path = 'Dataset'
    finger_dict = organize_dataset(dataset_path)
    pairs = create_pairs(finger_dict)
    data, labels = prepare_data(pairs)

    input_shape = (128, 128, 1)
    siamese_network = build_siamese_network(input_shape)
    siamese_network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = siamese_network.fit(
        [data[0], data[1]], labels,
        validation_split=0.2,
        epochs=20,
        batch_size=32
    )

    siamese_network.save('fingerprint_matching_model.h5')
    print("Model trained and saved successfully!")