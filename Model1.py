#Demo model
import os
import shutil
from sklearn.model_selection import train_test_split

# def split_data(source_dir, train_dir, test_dir, test_size=0.2):
#     """
#     Splits data into train and test directories.
    
#     Parameters:
#     - source_dir: Path to the original data folder (e.g., "data").
#     - train_dir: Path to the training data folder (e.g., "data/train").
#     - test_dir: Path to the testing data folder (e.g., "data/test").
#     - test_size: Fraction of data to allocate to the test set (default: 0.2).
#     """
#     # Ensure train and test directories exist
#     os.makedirs(train_dir, exist_ok=True)
#     os.makedirs(test_dir, exist_ok=True)

#     # Iterate through each class folder (opened, closed)
#     for class_name in os.listdir(source_dir):
#         class_path = os.path.join(source_dir, class_name)

#         # Skip if not a directory
#         if not os.path.isdir(class_path):
#             continue

#         # Get all file paths in the class folder
#         all_files = [os.path.join(class_path, f) for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]

#         # Split the data into train and test
#         train_files, test_files = train_test_split(all_files, test_size=test_size, random_state=42, shuffle=True)

#         # Create class folders in train and test directories
#         train_class_dir = os.path.join(train_dir, class_name)
#         test_class_dir = os.path.join(test_dir, class_name)
#         os.makedirs(train_class_dir, exist_ok=True)
#         os.makedirs(test_class_dir, exist_ok=True)

#         # Move files to the respective directories
#         for file in train_files:
#             shutil.copy(file, os.path.join(train_class_dir, os.path.basename(file)))
#         for file in test_files:
#             shutil.copy(file, os.path.join(test_class_dir, os.path.basename(file)))

#         print(f"Processed class '{class_name}': {len(train_files)} train, {len(test_files)} test")


# source_dir = "data"  # Original folder with images
# train_dir = "data/train"  # Destination folder for training data
# test_dir = "data/test"  # Destination folder for testing data

# # Split data
# split_data(source_dir, train_dir, test_dir, test_size=0.2)

train_dir = 'data/train'
test_dir = 'data/test'

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(rescale=1/255.,
                                # rotation_range=20,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                shear_range=0.2,
                                zoom_range=0.2,
                                # horizontal_flip=True,
                                fill_mode='nearest',
                                brightness_range=[0.8, 1.2],  # Random brightness change (between 80% and 120%)
                                channel_shift_range=10.0 )
test_gen = ImageDataGenerator(rescale=1/255.)

train_data = train_gen.flow_from_directory(directory = train_dir,
                                           target_size = (224,224),
                                           batch_size = 10,
                                           color_mode = "rgb",
                                           class_mode = "categorical",
                                           shuffle=True,
                                           seed = 42)

test_data = test_gen.flow_from_directory(directory = test_dir,
                                         target_size = (224,224),
                                         batch_size = 10,
                                         color_mode = "rgb",
                                         class_mode = "categorical",
                                         shuffle=False,
                                         seed = 42)


for batch_index, (images, labels) in enumerate(train_data):
    print(f"Batch {batch_index + 1}:")
    
    # Loop through all images in the current batch
    for i in range(len(images)):
        print(f"  Shape of image {i+1}: {images[i].shape}")
        print(f"  Label of image {i+1}: {labels[i]}")
    if batch_index == 39:  # Stop after 6 batches (index starts from 0)
        break

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2

# base_model = tf.keras.applications.efficientnet.EfficientNetB0(include_top= False, weights= "imagenet", input_shape= (224, 224, 3), pooling= 'max')
# # base_model.trainable = False

# model = Sequential([
#     base_model,
#     BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001),
#     Dense(12, kernel_regularizer= regularizers.l2(0.016), activity_regularizer= regularizers.l1(0.006),
#                 bias_regularizer= regularizers.l1(0.006), activation= 'relu'),
#     Dropout(rate= 0.45, seed= 123),
#     Dense(2, activation= 'softmax')
# ])

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001), input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    MaxPooling2D((2, 2)),

    # Flatten the features
    Flatten(),
    # BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001),
    # # Dense layer with L2 regularization
    # Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
    # Dropout(0.5),  # Dropout for additional regularization

    # # # Output layer
    # Dense(2, activation='softmax', kernel_regularizer=l2(0.001))
])

# Compile the model
# model.compile(optimizer=Adamax(), loss='categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# history = model.fit(train_data, epochs=10, validation_data=test_data)

model.save('test_model.h5')