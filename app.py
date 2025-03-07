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

nltk.download('stopwords')
nltk.download('wordnet')

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load model and preprocessing tools safely
try:
    model = load_model("model.keras")
    label_encoder = joblib.load("label_encoder.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    logging.info("✅ Model and preprocessing tools loaded successfully.")
except Exception as e:
    logging.error(f"❌ Error loading model or preprocessing tools: {e}")
    raise RuntimeError("Failed to load model/vectorizer/label_encoder.")

def preprocess_text(text):
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    words = text.split()
    # Remove stopwords
    words = [word for word in words if word not in stopwords.words('english')]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symptoms = data.get("symptoms", "").strip()
        follow_up = data.get("follow_up", "").strip().lower() if data.get("follow_up") else ""

        if not symptoms:
            return jsonify({"error": "No symptoms provided"}), 400

        symptoms = preprocess_text(symptoms)
        symptoms_vectorized = vectorizer.transform([symptoms]).toarray()
        prediction = model.predict(symptoms_vectorized).argmax(axis=1)
        illness = label_encoder.inverse_transform(prediction)
        diagnosis_message = f"The predicted illness is: {illness[0]}"
        
        if not follow_up:
            if illness[0].lower() in follow_up_questions:
                follow_up_question = follow_up_questions[illness[0].lower()]
                return jsonify({"diagnosis_message": diagnosis_message, "follow_up_question": follow_up_question})
            else:
                for symptom, questions in detailed_follow_up.items():
                    if symptom in symptoms:
                        for condition, question in questions:
                            return jsonify({"diagnosis_message": diagnosis_message, "follow_up_question": question})
                return jsonify({"diagnosis_message": diagnosis_message})
        else:
            if follow_up == "yes":
                diagnosis_message += f"\nThe diagnosis of {illness[0]} is more likely."
            else:
                diagnosis_message += f"\nThe diagnosis of {illness[0]} is less likely. Please consult a doctor for a more accurate diagnosis."
        
        return jsonify({"diagnosis_message": diagnosis_message})

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
