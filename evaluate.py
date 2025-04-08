from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import pandas as pd
import logging

def show_class_mapping(label_encoder):
    """Displays the mapping between class indices and illness names."""
    print("\n--- Class to Illness Mapping ---")
    mapping = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))
    # Print in a more readable format, perhaps sorted by index
    for index, illness in sorted(mapping.items()):
        print(f"Class {index}: {illness}")
    print("------------------------------")

def evaluate_model(model, X_test, y_test, label_encoder, X_train=None, y_train=None, is_keras_model=True):
    """
    Evaluates a trained model (Keras or scikit-learn compatible).

    Args:
        model: The trained model object.
        X_test (np.array): Test features.
        y_test (np.array): True test labels (encoded).
        label_encoder (LabelEncoder): Fitted label encoder for class names.
        X_train (np.array, optional): Training features for overfitting check. Defaults to None.
        y_train (np.array, optional): True training labels (encoded) for overfitting check. Defaults to None.
        is_keras_model (bool): Flag indicating if the model is a Keras model (requires .predict and .argmax). Defaults to True.
    """
    logging.info("Starting model evaluation...")

    # Make predictions
    if is_keras_model:
        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
    else: # Assume scikit-learn style .predict
        y_pred = model.predict(X_test)
        try:
            y_pred_probs = model.predict_proba(X_test)
        except AttributeError:
            y_pred_probs = None # Model doesn't support predict_proba

    # Display class mapping
    if label_encoder:
        show_class_mapping(label_encoder)

    # --- Overall Metrics ---
    test_acc = accuracy_score(y_test, y_pred)
    print(f"\n--- Overall Performance ---")
    print(f"Test Accuracy: {test_acc:.4f}")

    # --- Overfitting Check ---
    if X_train is not None and y_train is not None:
        if is_keras_model:
            train_pred_probs = model.predict(X_train)
            train_pred = np.argmax(train_pred_probs, axis=1)
        else:
             train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        print(f"Train Accuracy: {train_acc:.4f}")
        diff = abs(train_acc - test_acc)
        print(f"Difference (Train - Test): {diff:.4f}")
        if diff > 0.15: # Adjusted threshold
            logging.warning(f"Potential Overfitting Detected: Difference > 15% ({diff:.2%})")
        elif diff < -0.05:
            logging.warning(f"Potential Underfitting or Data Issue: Test accuracy significantly higher than train ({diff:.2%})")

    # --- Prediction Confidence (Keras only) ---
    if is_keras_model and y_pred_probs is not None:
        confidence = np.max(y_pred_probs, axis=1)
        confidence_mean = np.mean(confidence)
        confidence_std = np.std(confidence)
        print(f"\n--- Prediction Confidence ---")
        print(f"Mean Max Probability: {confidence_mean:.4f}")
        print(f"Std Dev Max Probability: {confidence_std:.4f}")
        print(f"Min Max Probability: {np.min(confidence):.4f}")
        print(f"Max Max Probability: {np.max(confidence):.4f}")

    # --- Classification Report ---
    print("\n--- Classification Report ---")
    # Get unique labels present in y_test OR y_pred to avoid errors if some classes are missing in test
    present_labels = np.unique(np.concatenate((y_test, y_pred)))
    target_names = label_encoder.inverse_transform(present_labels) if label_encoder else None
    try:
        report = classification_report(y_test, y_pred, labels=present_labels, target_names=target_names, zero_division=0)
        print(report)
    except Exception as e:
        logging.error(f"Could not generate classification report: {e}")
        # Fallback: print basic report without names
        report = classification_report(y_test, y_pred, labels=present_labels, zero_division=0)
        print(report)


    # --- Confusion Matrix ---
    print("\n--- Confusion Matrix ---")
    try:
        matrix = confusion_matrix(y_test, y_pred, labels=present_labels)
        if label_encoder:
            # Display as DataFrame for better readability if many classes
            if len(present_labels) > 15:
                 # Print non-zero elements for large matrices
                print("Non-zero entries (Actual -> Predicted: Count):")
                for i, j in zip(*np.nonzero(matrix)):
                     actual_label_idx = present_labels[i]
                     predicted_label_idx = present_labels[j]
                     actual_name = label_encoder.inverse_transform([actual_label_idx])[0]
                     predicted_name = label_encoder.inverse_transform([predicted_label_idx])[0]
                     count = matrix[i, j]
                     print(f"  {actual_name} -> {predicted_name}: {count}")
            else:
                # Use DataFrame for smaller matrices
                cm_df = pd.DataFrame(matrix, index=target_names, columns=target_names)
                print(cm_df)
        else:
            print(matrix) # Print raw matrix if no encoder
    except Exception as e:
        logging.error(f"Could not generate confusion matrix: {e}")

    print("---------------------------\n")
    logging.info("Model evaluation finished.")