from data_preparation import load_and_preprocess_data
from model import build_model, train_model
from evaluate import evaluate_model
from follow_up_questions import follow_up_questions, detailed_follow_up
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
import spacy
import os

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def normalize_symptoms(symptoms):
    # Use spaCy for advanced text processing
    doc = nlp(symptoms)
    words = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    normalized_symptoms = ' '.join(words)
    return normalized_symptoms

def get_user_input(vectorizer, label_encoder, model):
    symptoms = input("Enter your symptoms separated by commas: ").lower()
    symptoms = normalize_symptoms(symptoms)
    symptoms_vectorized = vectorizer.transform([symptoms]).toarray()
    prediction = model.predict(symptoms_vectorized)
    confidence = np.max(prediction)
    illness = label_encoder.inverse_transform(prediction.argmax(axis=1))
    print(f"The predicted illness is: {illness[0]} with confidence {confidence:.2f}")
    
    if illness[0].lower() in follow_up_questions:
        follow_up = input(follow_up_questions[illness[0].lower()] + " (yes/no): ").lower()
        if follow_up == "yes":
            print(f"The diagnosis of {illness[0]} is more likely.")
        else:
            print(f"The diagnosis of {illness[0]} is less likely. Please consult a doctor for a more accurate diagnosis.")
    else:
        for symptom, questions in detailed_follow_up.items():
            if symptom in symptoms:
                for condition, question in questions:
                    follow_up = input(question + " (yes/no): ").lower()
                    if follow_up == "yes":
                        print(f"The diagnosis of {condition} is more likely.")
                        break
                else:
                    print("The diagnosis is inconclusive. Please consult a doctor for a more accurate diagnosis.")
                break
        else:
            print("The diagnosis is inconclusive. Please consult a doctor for a more accurate diagnosis.")

def ensemble_predictions(models, X):
    if not models:
        raise ValueError("No models available for prediction")
        
    # Get number of classes from the last layer's config
    num_classes = models[0].layers[-1].get_config()['units']
    predictions = np.zeros((X.shape[0], num_classes))
    for model in models:
        predictions += model.predict(X)
    return predictions / len(models)

def train_model_with_default_progress(model, X_train, y_train, epochs=150, batch_size=32):
    import tensorflow as tf
    
    # Create standard callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            min_delta=0.0005
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.00001
        )
    ]
    
    print("Starting model training...")
    
    # Train the model with default progress bar
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=callbacks,
        shuffle=True,
        verbose=1  # Default progress bar
    )
    
    return history

def main():
    # Load and preprocess the data
    X, X_test, y, y_test, label_encoder, vectorizer = load_and_preprocess_data(r'C:\Users\stefe\Documents\GitHub\eminet\illnesses_and_symptoms.csv')
    
    # Convert y to a NumPy array
    y = np.array(y)
    
    # Cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    models = []
    
    # Track fold performances
    fold_performances = []
    
    for fold, (train_index, val_index) in enumerate(kfold.split(X)):
        print(f"\nTraining fold {fold + 1}/5")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        try:
            # Build model
            num_classes = len(label_encoder.classes_)
            model = build_model(X_train.shape[1], num_classes)
            
            # Train model with default progress bar
            history = train_model_with_default_progress(model, X_train, y_train)
            
            # Evaluate with comprehensive metrics
            print(f"\nFold {fold + 1} Evaluation:")
            evaluate_model(model, X_val, y_val, X_train, y_train, label_encoder)
            
            # Store model and performance
            val_loss, val_accuracy = model.evaluate(X_val, y_val)
            fold_performances.append(val_accuracy)
            models.append(model)
        except Exception as e:
            print(f"Error during fold {fold+1} training: {e}")
            continue
    
    # Handle case where no models were successfully trained
    if not models:
        print("\nError: No models were successfully trained. Please check your data and model configuration.")
        return
    
    print("\nCross-validation performance:")
    print(f"Mean accuracy: {np.mean(fold_performances):.4f}")
    print(f"Std deviation: {np.std(fold_performances):.4f}")
    
    # Final evaluation on test set
    print("\nFinal Evaluation on Test Set:")
    try:
        predictions = ensemble_predictions(models, X_test)
        y_pred = predictions.argmax(axis=1)
        evaluate_model(models[0], X_test, y_test)
        
        # Save best model
        best_model_index = np.argmax(fold_performances)
        models[best_model_index].save('model.keras')
        joblib.dump(label_encoder, 'label_encoder.pkl')
        joblib.dump(vectorizer, 'vectorizer.pkl')
    except Exception as e:
        print(f"Error during final evaluation: {e}")
        # Try to save a model if any are available
        if models:
            print("Saving the first successful model.")
            models[0].save('model.keras')
            joblib.dump(label_encoder, 'label_encoder.pkl')
            joblib.dump(vectorizer, 'vectorizer.pkl')

if __name__ == "__main__":
    import tensorflow as tf
    main()
