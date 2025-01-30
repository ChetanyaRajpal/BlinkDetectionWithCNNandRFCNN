import numpy as np
from tensorflow.keras.models import load_model, Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input

# Step 1: Load the pre-trained model
from tensorflow.keras.layers import Input


# Step 1: Define the input layer explicitly
input_layer = Input(shape=(224, 224, 3))

# Step 2: Define the rest of the model layers
x = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(input_layer)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(x)
x = MaxPooling2D((2, 2))(x)

# Flatten the features
x = Flatten()(x)

# Create the model
model = Model(inputs=input_layer, outputs=x)

# Step 3: Define the feature extractor
feature_extractor = Model(inputs=model.input, outputs=model.get_layer("flatten").output)

# Verify the model structure
feature_extractor.summary()


# Step 3: Feature extraction function
def extract_features(generator, feature_extractor):
    """
    Extracts features and labels from a data generator using the CNN feature extractor.
    
    Parameters:
    - generator: Data generator (train/test data).
    - feature_extractor: CNN truncated to a specific layer for feature extraction.
    
    Returns:
    - features: NumPy array of extracted features.
    - labels: Corresponding labels in a 1D array.
    """
    features = []
    labels = []
    for batch_images, batch_labels in generator:
        # Pass images through the feature extractor
        extracted_features = feature_extractor.predict(batch_images)
        features.append(extracted_features)
        labels.append(batch_labels)
        
        # Stop when all samples are processed
        if len(features) * generator.batch_size >= generator.samples:
            break
    # Stack all features and convert labels from one-hot to integer
    return np.vstack(features), np.argmax(np.vstack(labels), axis=1)

# Step 4: Prepare data generators
train_gen = ImageDataGenerator(rescale=1/255.,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                shear_range=0.2,
                                zoom_range=0.2,
                                fill_mode='nearest',
                                brightness_range=[0.8, 1.2],
                                channel_shift_range=10.0)

test_gen = ImageDataGenerator(rescale=1/255.)

train_dir = "data/train"
test_dir = "data/test"

train_data = train_gen.flow_from_directory(directory=train_dir,
                                           target_size=(224, 224),
                                           batch_size=10,
                                           color_mode="rgb",
                                           class_mode="categorical",
                                           shuffle=True,
                                           seed=42)

test_data = test_gen.flow_from_directory(directory=test_dir,
                                         target_size=(224, 224),
                                         batch_size=10,
                                         color_mode="rgb",
                                         class_mode="categorical",
                                         shuffle=False,
                                         seed=42)

# Step 5: Extract features and labels
X_train, y_train = extract_features(train_data, feature_extractor)
X_test, y_test = extract_features(test_data, feature_extractor)

# Step 6: Train the Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Step 7: Make predictions and evaluate the model
y_pred = rf_clf.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy Score:", accuracy_score(y_test, y_pred))

import pickle
# Save the trained Random Forest model
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_clf, f)
