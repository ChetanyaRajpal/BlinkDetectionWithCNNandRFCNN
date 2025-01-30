import tensorflow as tf
import pickle
from sklearn.metrics import accuracy_score, classification_report
with open('random_forest_model.pkl', 'rb') as f:
    rf_clf_loaded = pickle.load(f)
# rf_clf_loaded.summary()
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import Model2
test_dir = 'data/test2'
test_gen = ImageDataGenerator(rescale=1/255.)
test_data = test_gen.flow_from_directory(directory = test_dir,
                                         target_size = (224,224),
                                         batch_size = 2,
                                         color_mode = "rgb",
                                         class_mode = "categorical",
                                         shuffle=True,
                                         seed = 42)

X_test, y_test = Model2.extract_features(test_data, Model2.feature_extractor)
# Step 2: Make predictions on the test data (X_test)
y_pred_loaded = rf_clf_loaded.predict(X_test)

# Step 3: Evaluate the model's performance
print("Classification Report:")
print(classification_report(y_test, y_pred_loaded))

print("Accuracy Score:", accuracy_score(y_test, y_pred_loaded))