from data_preparation import load_and_preprocess_data
from model import build_model, train_model
from evaluate import evaluate_model
from follow_up_questions import follow_up_questions, detailed_follow_up
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold

def normalize_symptoms(symptoms):
    normalization_dict = {
        "balance issues": "balance problems",
        "balance problems": "balance problems",
        # Add more normalization rules as needed
    }
    normalized_symptoms = []
    for symptom in symptoms.split(','):
        symptom = symptom.strip()
        normalized_symptoms.append(normalization_dict.get(symptom, symptom))
    return ' '.join(normalized_symptoms)

def get_user_input(vectorizer, label_encoder, model):
    symptoms = input("Enter your symptoms separated by commas: ").lower()
    symptoms = normalize_symptoms(symptoms)
    symptoms_vectorized = vectorizer.transform([symptoms]).toarray()
    prediction = model.predict(symptoms_vectorized).argmax(axis=1)
    illness = label_encoder.inverse_transform(prediction)
    print(f"The predicted illness is: {illness[0]}")
    
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

def main():
    # Load and preprocess the data
    X, X_test, y, y_test, label_encoder, vectorizer = load_and_preprocess_data(r'C:\Users\stefe\Documents\GitHub\eminet\illnesses_and_symptoms.csv')
    
    # Debugging: Print class distribution and data shapes
    unique, counts = np.unique(y, return_counts=True)
    print("Class distribution before splitting:", dict(zip(unique, counts)))
    print("X shape:", X.shape)
    print("X_test shape:", X_test.shape)
    print("y shape:", y.shape)
    print("y_test shape:", y_test.shape)
    
    # Debugging: Print samples of the preprocessed data and labels
    print("Sample preprocessed data:", X[:5])
    print("Sample labels:", y[:5])
    
    # Convert y to a NumPy array
    y = np.array(y)
    
    # Cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, val_index in kfold.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # Debugging: Print class distribution in training and validation sets
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        unique_val, counts_val = np.unique(y_val, return_counts=True)
        print("Class distribution in training set:", dict(zip(unique_train, counts_train)))
        print("Class distribution in validation set:", dict(zip(unique_val, counts_val)))
        
        # Build the model
        num_classes = len(label_encoder.classes_)
        model = build_model(X_train.shape[1], num_classes)
        
        # Train the model
        history = train_model(model, X_train, y_train, epochs=75, batch_size=32)
        
        # Debugging: Print training history
        print("Training history:", history.history)
        
        # Evaluate the model on validation data
        val_loss, val_accuracy = model.evaluate(X_val, y_val)
        print(f"Validation loss: {val_loss}, Validation accuracy: {val_accuracy}")
    
    # Evaluate the model on test data
    evaluate_model(model, X_test, y_test)
    
    # Save the model and preprocessing tools
    model.save('model.keras')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

if __name__ == "__main__":
    main()
