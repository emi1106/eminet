from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import logging
import re
import nltk
# Note: NLTK data download should be done manually once after installation
# See README.md
from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer # Using spaCy
from follow_up_questions import get_follow_up_question, get_detailed_follow_up
import spacy
import os
from translations import get_diagnosis_message, translate_medical_term, translate_illness

# =============================================================================
# Configuration & Initialization
# =============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Global variables for model and tools
model = None
label_encoder = None
vectorizer = None
nlp = None # spaCy language model

# =============================================================================
# Load Models and Preprocessing Tools
# =============================================================================
def load_resources():
    global model, label_encoder, vectorizer, nlp
    try:
        logging.info("Loading Keras model from model.keras...")
        model = load_model("model.keras")
        logging.info("Loading label encoder from label_encoder.pkl...")
        label_encoder = joblib.load("label_encoder.pkl")
        logging.info("Loading vectorizer from vectorizer.pkl...")
        vectorizer = joblib.load("vectorizer.pkl")
        logging.info("✅ Model and preprocessing tools loaded successfully.")
    except Exception as e:
        logging.error(f"❌ Error loading ML model or tools: {e}")
        model = label_encoder = vectorizer = None # Ensure all are None if any fails

    # Load spaCy model separately for NLP tasks
    try:
        logging.info("Loading spaCy model 'en_core_web_sm'...")
        nlp = spacy.load("en_core_web_sm")
        logging.info("✅ spaCy model loaded successfully.")
    except Exception as e:
        logging.error(f"❌ Error loading spaCy model: {e}")
        nlp = None
        # Attempt download if it seems like a "not found" error
        if "not found" in str(e).lower() or "can't find" in str(e).lower():
            logging.warning("spaCy model not found. Attempting download...")
            try:
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                nlp = spacy.load("en_core_web_sm")
                logging.info("✅ spaCy model downloaded and loaded successfully.")
            except Exception as e2:
                logging.error(f"❌ Failed to download/load spaCy model after attempt: {e2}")
                nlp = None
        if nlp is None:
             logging.warning("⚠️ spaCy model unavailable. Using basic text processing.")

# Load resources when the script starts
load_resources()

# =============================================================================
# Text Processing and Translation Functions
# =============================================================================
# Use a basic stopword list as a fallback if NLTK data isn't available
try:
    NLTK_STOPWORDS = set(stopwords.words('english'))
except LookupError:
    logging.warning("NLTK stopwords not found. Using a basic list. Run nltk.download('stopwords')")
    NLTK_STOPWORDS = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'}


def preprocess_text_for_api(text):
    """
    Process and normalize text input specifically for API usage.
    Uses spaCy if available, otherwise falls back to basic cleaning.
    """
    if not isinstance(text, str):
        text = str(text) # Ensure string input

    text = text.lower()

    if nlp:
        # Use spaCy for advanced processing
        doc = nlp(text)
        # Lemmatize, remove stopwords and non-alphabetic tokens
        words = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
        processed = ' '.join(words)
        # logging.debug(f"spaCy processed: '{text}' -> '{processed}'")
        return processed
    else:
        # Fallback to basic preprocessing
        # Remove punctuation and numbers
        text = re.sub(r'[^a-z\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Basic stopword removal
        words = [w for w in text.split() if w not in NLTK_STOPWORDS]
        processed = ' '.join(words)
        # logging.debug(f"Basic processed: '{text}' -> '{processed}'")
        return processed


def translate_symptoms_api(symptoms, language):
    """
    Translate symptoms for the API. Focuses on Romanian to English.
    Uses a dictionary for common terms and falls back to term-by-term translation.
    """
    if language == 'ro':
        # Quick lookup dictionary for common Romanian terms
        ro_en_symptoms_dict = {
            'febra': 'fever', 'febră': 'fever', 'durere': 'pain', 'dureri': 'pain',
            'cap': 'head', 'stomac': 'stomach', 'gat': 'throat', 'gât': 'throat',
            'tuse': 'cough', 'greață': 'nausea', 'greata': 'nausea',
            'vărsături': 'vomiting', 'varsaturi': 'vomiting', 'vomitat': 'vomiting',
            'diaree': 'diarrhea', 'constipație': 'constipation', 'constipatie': 'constipation',
            'amețeală': 'dizziness', 'ameteala': 'dizziness', 'oboseală': 'fatigue',
            'oboseala': 'fatigue', 'slăbiciune': 'weakness', 'slabiciune': 'weakness',
            'mâncărime': 'itching', 'mancarime': 'itching', 'erupție': 'rash', 'eruptie': 'rash',
            'respirație': 'breathing', 'respiratie': 'breathing', 'dificultate': 'difficulty',
            'sângerare': 'bleeding', 'sangerare': 'bleeding', 'vedere': 'vision', 'auz': 'hearing',
            'piept': 'chest', 'spate': 'back', 'picioare': 'legs', 'brațe': 'arms', 'brate': 'arms',
            'mâini': 'hands', 'maini': 'hands', 'ochi': 'eyes', 'urechi': 'ears'
            # Add more common terms as needed
        }

        # Clean input: lower, remove punctuation except spaces between words
        cleaned_symptoms = re.sub(r'[^\w\s]', '', symptoms.lower())
        words = cleaned_symptoms.split()

        translated_words = []
        for word in words:
            if word in ro_en_symptoms_dict:
                translated_words.append(ro_en_symptoms_dict[word])
            else:
                # Fallback to the more general translator function
                translated_term = translate_medical_term(word, to_romanian=False)
                # Avoid adding the original word if translation fails/is the same
                if translated_term != word:
                    translated_words.append(translated_term)
                # Optionally, keep the original word if no translation found
                # else:
                #    translated_words.append(word) # Keep original if no translation

        # Join translated words, removing duplicates while preserving order
        seen = set()
        unique_translated = [x for x in translated_words if not (x in seen or seen.add(x))]
        translated_text = ' '.join(unique_translated)

        logging.info(f"Translated RO -> EN: '{symptoms}' -> '{translated_text}'")
        return translated_text
    else:
        # Assume English or other unsupported language, return as is
        return symptoms

def get_disclaimer(language):
    """Return a medical disclaimer in the specified language."""
    if language == "ro":
        return "\n\nAVERTISMENT: Acesta este un rezultat generat de AI și nu înlocuiește consultul medical. Adresați-vă întotdeauna unui medic pentru diagnostic și tratament."
    else:  # Default to English
        return "\n\nDISCLAIMER: This is an AI-generated result and not a substitute for professional medical advice. Always consult a doctor for diagnosis and treatment."

# =============================================================================
# API Endpoints
# =============================================================================
@app.route('/health', methods=['GET'])
def health_check():
    """API endpoint to check system health and model status."""
    status = {
        "status": "ok" if model and vectorizer and label_encoder else "degraded",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "label_encoder_loaded": label_encoder is not None,
        "spacy_loaded": nlp is not None
    }
    return jsonify(status)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles symptom prediction requests."""
    # Check if model and tools are loaded
    if not all([model, vectorizer, label_encoder]):
        error_msg_en = "Service Unavailable: Model or necessary components are not loaded."
        error_msg_ro = "Serviciu Indisponibil: Modelul sau componentele necesare nu sunt încărcate."
        # Try to determine language from request if possible, else default
        lang = request.get_json().get("language", "en") if request.is_json else "en"
        return jsonify({
            "error": error_msg_ro if lang == "ro" else error_msg_en,
            "diagnosis_message": error_msg_ro if lang == "ro" else error_msg_en,
            "follow_up_question": ""
        }), 503 # Service Unavailable

    try:
        data = request.get_json()
        if not data:
             return jsonify({"error": "Invalid input: No JSON data received."}), 400

        symptoms_original = data.get("symptoms", "").strip()
        follow_up_answer = data.get("follow_up", "").strip().lower()
        language = data.get("language", "en")

        if not symptoms_original:
            error_msg = "No symptoms provided." if language == "en" else "Nu au fost furnizate simptome."
            return jsonify({"error": error_msg}), 400

        # --- Step 1: Translate and Preprocess Symptoms ---
        symptoms_english = translate_symptoms_api(symptoms_original, language)
        if not symptoms_english: # Check if translation resulted in empty string
             logging.warning(f"Translation/original symptoms resulted in empty string for input: {symptoms_original}")
             error_msg = "Could not process symptoms." if language == "en" else "Simptomele nu au putut fi procesate."
             return jsonify({"error": error_msg}), 400

        processed_symptoms = preprocess_text_for_api(symptoms_english)
        if not processed_symptoms: # Check if preprocessing resulted in empty string
            logging.warning(f"Preprocessing resulted in empty string for input: {symptoms_english}")
            error_msg = "Could not process symptoms after cleaning." if language == "en" else "Simptomele nu au putut fi procesate după curățare."
            return jsonify({"error": error_msg}), 400

        logging.info(f"Original: '{symptoms_original}' ({language}) -> Translated: '{symptoms_english}' -> Processed: '{processed_symptoms}'")

        # --- Step 2: Vectorize and Predict ---
        try:
             symptoms_vectorized = vectorizer.transform([processed_symptoms]).toarray()
        except Exception as e:
             logging.error(f"Error during vectorization: {e} for input '{processed_symptoms}'")
             error_msg = "Error processing symptoms." if language == "en" else "Eroare la procesarea simptomelor."
             return jsonify({"error": error_msg}), 500

        prediction_probs = model.predict(symptoms_vectorized)[0] # Get probabilities for the single input
        confidence = np.max(prediction_probs)
        illness_index = np.argmax(prediction_probs)
        illness_english = label_encoder.inverse_transform([illness_index])[0]

        # --- Step 3: Translate Results and Format Message ---
        illness_translated = translate_illness(illness_english, language)
        logging.info(f"Prediction: {illness_english} ({illness_translated}) with confidence {confidence:.2f}")

        # Get base diagnosis message
        diagnosis_message = get_diagnosis_message(
            language,
            "prediction",
            illness=illness_translated,
            confidence=confidence
        )

        follow_up_question_to_ask = ""

        # --- Step 4: Handle Follow-up Logic ---
        if follow_up_answer in ["yes", "da", "y"]:
            # User answered YES to a previous follow-up
            additional_message = get_diagnosis_message(language, "more_likely", illness=illness_translated)
            diagnosis_message += f"\n{additional_message}" # Append confirmation
        elif follow_up_answer in ["no", "nu", "n"]:
             # User answered NO to a previous follow-up
            additional_message = get_diagnosis_message(language, "less_likely", illness=illness_translated)
            diagnosis_message += f"\n{additional_message}" # Append note about being less likely
        else:
            # No valid follow-up answer provided, check if we need to ask a question
            # Prioritize illness-specific question
            follow_up_question_to_ask = get_follow_up_question(illness_english.lower(), language)

            # If no illness-specific question, check for symptom-specific detailed questions
            if not follow_up_question_to_ask:
                symptom_words = processed_symptoms.split()
                for symptom in symptom_words:
                    detailed_questions = get_detailed_follow_up(symptom, language)
                    if detailed_questions:
                        # Ask the first relevant detailed question found
                        # (Could potentially be refined to pick the 'best' one)
                        condition_en, question_text = detailed_questions[0]
                        follow_up_question_to_ask = question_text
                        logging.info(f"Asking detailed follow-up for symptom '{symptom}': {question_text}")
                        break # Stop after finding the first detailed question

        # --- Step 5: Add Disclaimer and Return Response ---
        final_diagnosis_message = diagnosis_message + get_disclaimer(language)

        return jsonify({
            "diagnosis_message": final_diagnosis_message,
            "follow_up_question": follow_up_question_to_ask
            # Optionally return confidence, english illness name etc.
            # "confidence": float(confidence),
            # "illness_en": illness_english
        })

    except Exception as e:
        logging.exception(f"An unexpected error occurred during prediction: {e}") # Log full traceback
        error_msg = "Internal server error." if language == "en" else "Eroare internă de server."
        return jsonify({"error": error_msg, "diagnosis_message": error_msg, "follow_up_question": ""}), 500

# =============================================================================
# Run Flask App
# =============================================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # Use FLASK_DEBUG environment variable to control debug mode
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')

    logging.info(f"⚡ Starting Flask server on port {port} with debug={debug_mode}")
    # Consider using a production server like gunicorn instead of app.run in production
    app.run(debug=debug_mode, host="0.0.0.0", port=port)