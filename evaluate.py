from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import numpy as np

def show_class_mapping(label_encoder):
    """Display mapping between class numbers and illnesses"""
    print("\nClass to Illness Mapping:")
    for i, illness in enumerate(label_encoder.classes_):
        print(f"Class {i}: {illness}")

def evaluate_model(model, X_test, y_test, X_train=None, y_train=None, label_encoder=None):
    # Standard evaluation
    y_pred = model.predict(X_test).argmax(axis=1)
    
    # Show class mapping if label_encoder is provided
    if label_encoder is not None:
        show_class_mapping(label_encoder)
    
    # Get detailed metrics
    report = classification_report(y_test, y_pred, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred)
    
    # Additional evaluation metrics
    if X_train is not None and y_train is not None:
        # Check for overfitting by comparing train vs test performance
        train_pred = model.predict(X_train).argmax(axis=1)
        train_acc = (train_pred == y_train).mean()
        test_acc = (y_pred == y_test).mean()
        
        print(f"\nTrain accuracy: {train_acc:.4f}")
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Difference: {abs(train_acc - test_acc):.4f}")
        
        if abs(train_acc - test_acc) > 0.1:
            print("Warning: Model might be overfitting (>10% difference between train and test accuracy)")
    
    # Show prediction probabilities distribution
    probs = model.predict(X_test)
    confidence_mean = np.mean(np.max(probs, axis=1))
    confidence_std = np.std(np.max(probs, axis=1))
    print(f"\nMean prediction confidence: {confidence_mean:.4f}")
    print(f"Confidence std deviation: {confidence_std:.4f}")
    
    print("\nClassification Report:")
    print(report)
    
    # Print readable confusion matrix if label encoder is provided
    print("\nConfusion Matrix:")
    if label_encoder is not None:
        # Get unique classes present in test set and predictions
        classes = np.unique(np.concatenate([y_test, y_pred]))
        illnesses = label_encoder.inverse_transform(classes)
        
        # Print header
        print("\nPredicted vs Actual Illnesses (non-zero entries only):")
        for i, j in zip(*np.nonzero(matrix)):
            if matrix[i, j] > 0:
                actual = label_encoder.inverse_transform([i])[0]
                predicted = label_encoder.inverse_transform([j])[0]
                count = matrix[i, j]
                print(f"Actual: {actual:20} | Predicted: {predicted:20} | Count: {count}")
    else:
        print(matrix)
