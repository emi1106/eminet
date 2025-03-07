import joblib
from tensorflow.keras.models import load_model
from main import normalize_symptoms, get_user_input

def main():
    # Load the model and preprocessing tools
    model = load_model('model.keras')
    label_encoder = joblib.load('label_encoder.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    # Get user input and make a prediction
    get_user_input(vectorizer, label_encoder, model)

if __name__ == "__main__":
    main()
