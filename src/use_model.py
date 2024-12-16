import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from data_preparation import prepare_data

def evaluate_model(model_path, test_generator):
    model = load_model(model_path)
    print(f"Model loaded from {model_path}")
    
    loss, accuracy = model.evaluate(test_generator)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)
    
    predictions = np.argmax(model.predict(test_generator), axis=-1)
    true_labels = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())
    
    cm = confusion_matrix(true_labels, predictions)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(true_labels, predictions, target_names=class_labels)) 