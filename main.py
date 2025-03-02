from data_preparation import load_and_preprocess_data
from model import build_model, train_model
from evaluate import evaluate_model
import numpy as np

def get_user_input(vectorizer, label_encoder, model):
    symptoms = input("Enter your symptoms separated by commas: ").lower()
    symptoms = ' '.join(symptoms.split(','))
    symptoms_vectorized = vectorizer.transform([symptoms]).toarray()
    prediction = model.predict(symptoms_vectorized).argmax(axis=1)
    illness = label_encoder.inverse_transform(prediction)
    print(f"The predicted illness is: {illness[0]}")

def main():
    # Load and preprocess the data
    X_train, X_test, y_train, y_test, label_encoder, vectorizer = load_and_preprocess_data(r'C:\Users\stefe\Documents\GitHub\eminet\illnesses_and_symptoms.csv')
    
    # Build the model
    num_classes = len(label_encoder.classes_)
    model = build_model(X_train.shape[1], num_classes)
    
    # Train the model
    train_model(model, X_train, y_train, epochs=200)  # Increase the number of epochs
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)
    
    # Get user input and make predictions
    get_user_input(vectorizer, label_encoder, model)

if __name__ == "__main__":
    main()
