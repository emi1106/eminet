import joblib
from tensorflow.keras.models import load_model
import numpy as np
import logging
import spacy # Needed for preprocessing

# Import necessary functions from project modules
from data_preparation import preprocess_text # Use the main preprocessing function
from follow_up_questions import get_follow_up_question, get_detailed_follow_up
from translations import translate_illness # To show translated illness if needed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load spaCy model globally for this script
try:
    nlp = spacy.load("en_core_web_sm")
    logging.info("spaCy model loaded for preprocessing.")
except OSError:
    logging.error("spaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'")
    nlp = None # Allow script to potentially fail later if preprocessing needed

MODEL_PATH = 'model.keras'
ENCODER_PATH = 'label_encoder.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'

def get_user_input_and_predict(vectorizer, label_encoder, model):
    """
    Handles interactive user input, preprocessing, prediction, and follow-up questions.
    """
    if not nlp:
        logging.error("spaCy model not loaded. Cannot preprocess text. Exiting.")
        return

    while True:
        try:
            symptoms_raw = input("Enter your symptoms separated by commas (or type 'quit' to exit): ").strip()
            if symptoms_raw.lower() == 'quit':
                break
            if not symptoms_raw:
                print("Please enter some symptoms.")
                continue

            # Preprocess the input symptoms using the same function as training
            symptoms_processed = preprocess_text(symptoms_raw)
            logging.info(f"Raw: '{symptoms_raw}' -> Processed: '{symptoms_processed}'")

            if not symptoms_processed:
                print("Could not understand the symptoms after processing. Please try rephrasing.")
                continue

            # Vectorize the processed symptoms
            try:
                symptoms_vectorized = vectorizer.transform([symptoms_processed]).toarray()
            except Exception as e:
                 logging.error(f"Error vectorizing input: {e}")
                 print("An error occurred while processing symptoms.")
                 continue

            # Make prediction
            prediction_probs = model.predict(symptoms_vectorized)[0]
            confidence = np.max(prediction_probs)
            illness_index = np.argmax(prediction_probs)
            illness_en = label_encoder.inverse_transform([illness_index])[0]

            print(f"\nPredicted Illness: {illness_en} (Confidence: {confidence:.2%})")

            # --- Follow-up Question Logic ---
            follow_up_question = get_follow_up_question(illness_en.lower(), language="en") # Test in English

            if not follow_up_question:
                # Check detailed symptom questions if no illness-specific one
                symptom_words = symptoms_processed.split()
                for symptom in symptom_words:
                    detailed_questions = get_detailed_follow_up(symptom, language="en")
                    if detailed_questions:
                        condition_en, question_text = detailed_questions[0]
                        follow_up_question = question_text
                        logging.info(f"Asking detailed follow-up for symptom '{symptom}': {question_text}")
                        break # Ask first relevant detailed question

            if follow_up_question:
                follow_up_answer = input(f"{follow_up_question} (yes/no): ").strip().lower()
                if follow_up_answer == "yes":
                    print(f"-> Based on your answer, {illness_en} seems more likely.")
                elif follow_up_answer == "no":
                     print(f"-> Based on your answer, {illness_en} might be less likely, or another condition could be present.")
                else:
                    print("-> Answer not recognized.")
            else:
                print("(No specific follow-up question for this prediction.)")

            print("\n---\n") # Separator for next input

        except EOFError: # Handle Ctrl+D or end of input stream
             break
        except Exception as e:
            logging.exception(f"An error occurred during the loop: {e}")
            print("An unexpected error occurred. Please try again.")


def main():
    """Loads resources and starts the interactive prediction loop."""
    logging.info("Loading model and preprocessing tools for testing...")
    try:
        model = load_model(MODEL_PATH)
        label_encoder = joblib.load(ENCODER_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        logging.info("Model, encoder, and vectorizer loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load required files: {e}")
        print(f"Error: Could not load model or associated files ({MODEL_PATH}, {ENCODER_PATH}, {VECTORIZER_PATH}).")
        print("Please ensure the model has been trained using 'python main.py'.")
        return

    # Start interactive loop
    get_user_input_and_predict(vectorizer, label_encoder, model)

    logging.info("Exiting interactive test.")

if __name__ == "__main__":
    # Ensure NLTK data is available (user responsibility - see README)
    # Not strictly needed here if preprocess_text uses spaCy primarily,
    # but good practice to have it available if data_preparation relies on it.
    import nltk
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        logging.warning("NLTK WordNet data not found. Synonym features in preprocessing might be affected if used.")
        # No exit needed, as spaCy handles primary preprocessing here.

    main()