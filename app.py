from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import logging
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from follow_up_questions import follow_up_questions, detailed_follow_up
import spacy
import os
from translations import get_diagnosis_message, translate_medical_term, translate_illness

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load model and preprocessing tools safely
try:
    model = load_model("model.keras")
    label_encoder = joblib.load("label_encoder.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    logging.info("✅ Model and preprocessing tools loaded successfully.")
except Exception as e:
    logging.error(f"❌ Error loading model or preprocessing tools: {e}")
    model = None
    label_encoder = None
    vectorizer = None
    logging.warning("⚠️ App will run in degraded mode - model loading failed")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    logging.info("✅ spaCy model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Error loading spaCy model: {e}")
    # Try to download the model
    try:
        import subprocess
        logging.info("⏳ Attempting to download spaCy model...")
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        nlp = spacy.load("en_core_web_sm")
        logging.info("✅ spaCy model downloaded and loaded successfully.")
    except Exception as e2:
        logging.error(f"❌ Failed to download spaCy model: {e2}")
        nlp = None
        logging.warning("⚠️ App will use basic text processing instead of spaCy")

def translate_symptoms(symptoms, language):
    """
    Translate symptoms between languages if needed.
    For Romanian to English (model's language), we need to translate.
    """
    if language == 'ro':
        # Common Romanian symptoms dictionary for direct translation
        ro_en_symptoms = {
            'febra': 'fever',
            'febră': 'fever',
            'durere': 'pain',
            'dureri': 'pain',
            'cap': 'head',
            'stomac': 'stomach',
            'gat': 'throat',
            'gât': 'throat',
            'tuse': 'cough',
            'greață': 'nausea',
            'greata': 'nausea',
            'vărsături': 'vomiting',
            'varsaturi': 'vomiting',
            'vomitat': 'vomiting',
            'diaree': 'diarrhea',
            'constipație': 'constipation',
            'constipatie': 'constipation',
            'amețeală': 'dizziness',
            'ameteala': 'dizziness',
            'oboseală': 'fatigue',
            'oboseala': 'fatigue',
            'slăbiciune': 'weakness',
            'slabiciune': 'weakness',
            'mâncărime': 'itching',
            'mancarime': 'itching',
            'erupție': 'rash',
            'eruptie': 'rash',
            'respirație': 'breathing',
            'respiratie': 'breathing',
            'dificultate': 'difficulty',
            'sângerare': 'bleeding',
            'sangerare': 'bleeding',
            'vedere': 'vision',
            'auz': 'hearing',
            'piept': 'chest',
            'spate': 'back',
            'picioare': 'legs',
            'brațe': 'arms',
            'brate': 'arms',
            'mâini': 'hands',
            'maini': 'hands',
            'ochi': 'eyes',
            'urechi': 'ears'
        }
        
        # Clean and split the input, handling punctuation
        cleaned_symptoms = re.sub(r'[,;]', ' ', symptoms.lower())
        words = cleaned_symptoms.split()
        
        # Translate each word, checking the dictionary first
        translated_words = []
        for word in words:
            # Remove any remaining punctuation for lookup
            clean_word = re.sub(r'[^\w\s]', '', word)
            if clean_word in ro_en_symptoms:
                translated_words.append(ro_en_symptoms[clean_word])
            else:
                # Fall back to the function-based translation
                translated_words.append(translate_medical_term(clean_word, to_romanian=False))
        
        translated_text = ' '.join(translated_words)
        logging.info(f"Translated from Romanian: '{symptoms}' -> '{translated_text}'")
        return translated_text
    
    return symptoms  # Return as-is for English

def preprocess_text(text):
    """Process and normalize text input"""
    if nlp is not None:
        # Use spaCy for advanced text processing
        doc = nlp(text.lower())
        words = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        return ' '.join(words)
    else:
        # Fallback to basic preprocessing
        text = text.lower()
        # Remove non-alphanumeric characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Basic stopword removal (if NLTK available)
        try:
            stop_words = set(stopwords.words('english'))
            words = [w for w in text.split() if w not in stop_words]
            return ' '.join(words)
        except:
            return text

@app.route('/health', methods=['GET'])
def health_check():
    """API endpoint to check system health"""
    status = {
        "status": "ok" if model is not None and vectorizer is not None and label_encoder is not None else "degraded",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "label_encoder_loaded": label_encoder is not None,
        "spacy_loaded": nlp is not None
    }
    return jsonify(status)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symptoms = data.get("symptoms", "").strip()
        follow_up = data.get("follow_up", "")
        language = data.get("language", "en")  # Default to English

        # Check if essential components are loaded
        if model is None or vectorizer is None or label_encoder is None:
            error_msg = "System not fully initialized. The model could not be loaded."
            if language == "ro":
                error_msg = "Sistemul nu este complet inițializat. Modelul nu a putut fi încărcat."
            return jsonify({
                "error": error_msg,
                "diagnosis_message": error_msg,
                "follow_up_question": ""
            }), 503

        if not symptoms:
            error_msg = "No symptoms provided"
            if language == "ro":
                error_msg = "Nu au fost furnizate simptome"
            return jsonify({"error": error_msg}), 400

        # Process symptoms based on language
        translated_symptoms = translate_symptoms(symptoms, language)
        processed_symptoms = preprocess_text(translated_symptoms)
        
        # Vectorize and predict
        symptoms_vectorized = vectorizer.transform([processed_symptoms]).toarray()
        prediction = model.predict(symptoms_vectorized)
        confidence = np.max(prediction)
        illness_index = prediction.argmax(axis=1)[0]
        illness = label_encoder.inverse_transform([illness_index])[0]
        
        # Translate illness if needed
        translated_illness = translate_illness(illness, language)
        
        # Get appropriate diagnosis message
        diagnosis_message = get_diagnosis_message(
            language, 
            "prediction", 
            illness=translated_illness, 
            confidence=confidence
        )

        # Debug: Log what is being returned
        logging.info(f"Original symptoms: {symptoms}, Translated: {translated_symptoms}")
        logging.info(f"Diagnosis: {illness} ({translated_illness}), Confidence: {confidence:.2f}")

        # Handle follow-up logic
        if not follow_up:
            # No follow-up provided yet, check if we need to ask one
            follow_up_question = ""
            
            # Get appropriate follow-up question based on the diagnosed illness
            if illness.lower() in follow_up_questions:
                follow_up_question = follow_up_questions[illness.lower()]
                # Translate follow-up question if Romanian is requested
                if language == "ro":
                    # Basic translation of common follow-up questions
                    if "Do you have" in follow_up_question:
                        follow_up_question = follow_up_question.replace("Do you have", "Aveți")
                    if "Have you experienced" in follow_up_question:
                        follow_up_question = follow_up_question.replace("Have you experienced", "Ați experimentat")
                    if "?" not in follow_up_question:
                        follow_up_question += "?"
                
                return jsonify({
                    "diagnosis_message": diagnosis_message, 
                    "follow_up_question": follow_up_question
                })
            else:
                # Check if any symptoms match detailed follow-up questions
                for symptom, questions in detailed_follow_up.items():
                    if symptom in processed_symptoms:
                        for condition, question in questions:
                            # Basic translation for Romanian
                            if language == "ro":
                                if "Do you have" in question:
                                    question = question.replace("Do you have", "Aveți")
                                if "Have you experienced" in question:
                                    question = question.replace("Have you experienced", "Ați experimentat")
                                if "?" not in question:
                                    question += "?"
                            
                            return jsonify({
                                "diagnosis_message": diagnosis_message, 
                                "follow_up_question": question
                            })
            
            # If no follow-up questions, return just the diagnosis
            return jsonify({
                "diagnosis_message": diagnosis_message, 
                "follow_up_question": ""
            })
        else:
            # Process the follow-up answer
            follow_up = follow_up.strip().lower()
            if follow_up in ["yes", "da", "y"]:
                additional_message = get_diagnosis_message(
                    language,
                    "more_likely",
                    illness=translated_illness
                )
                diagnosis_message += f"\n{additional_message}"
            else:
                additional_message = get_diagnosis_message(
                    language,
                    "less_likely",
                    illness=translated_illness
                )
                diagnosis_message += f"\n{additional_message}"

            return jsonify({
                "diagnosis_message": diagnosis_message, 
                "follow_up_question": ""
            })

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        error_msg = "Internal server error"
        if language == "ro":
            error_msg = "Eroare internă de server"
        return jsonify({"error": error_msg}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    logging.info(f"⚡ Starting server on port {port} with debug={debug_mode}")
    app.run(debug=debug_mode, host="0.0.0.0", port=port)
