# Symptom-Based Diagnosis Neural Network (Eminet)

This project implements a Neural Network using TensorFlow/Keras to predict potential illnesses based on user-provided symptoms. It includes data preprocessing, augmentation, model training with cross-validation, evaluation, and a Flask API for serving predictions with multilingual support (English/Romanian).

## Features

*   Symptom preprocessing using spaCy (lemmatization, stopword removal).
*   Data augmentation using synonym replacement (via NLTK WordNet).
*   Handling class imbalance using SMOTE (from imbalanced-learn).
*   Neural Network model (Dense layers, Dropout, Batch Normalization).
*   Training pipeline with Stratified K-Fold Cross-Validation.
*   Model evaluation using standard metrics (accuracy, classification report, confusion matrix).
*   Flask API endpoint (`/predict`) to serve predictions.
*   Multilingual support (English/Romanian) for symptoms, diagnoses, and follow-up questions.
*   Basic health check endpoint (`/health`).

## Project Structure
├── venv/ # Virtual environment (ignored by git)
├── pycache/ # Python cache (ignored by git)
├── model.keras # Saved trained Keras model
├── label_encoder.pkl # Saved label encoder
├── vectorizer.pkl # Saved TF-IDF vectorizer
├── app.py # Flask application for API
├── data_preparation.py # Data loading, cleaning, preprocessing, augmentation, SMOTE
├── evaluate.py # Model evaluation functions
├── follow_up_questions.py # Stores follow-up questions (EN/RO)
├── illnesses_and_symptoms.csv # Raw dataset
├── main.py # Main script for training and evaluation pipeline
├── model.py # Keras model definition and training function
├── requirements.txt # Python package dependencies
├── test_model.py # Script for interactive command-line testing
├── translations.py # Translation dictionaries and functions (EN/RO)
├── .gitignore # Git ignore file
└── README.md # This file
## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/eminet.git # Replace with your repo URL
    cd eminet
    ```

2.  **Create and Activate Virtual Environment:**
    ```bash
    # Create environment
    python -m venv venv

    # Activate environment
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Python Packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Required NLP Data:**
    *   **NLTK Data (WordNet, Stopwords):**
        ```bash
        python -m nltk.download wordnet
        python -m nltk.download stopwords
        ```
    *   **spaCy English Model:**
        ```bash
        python -m spacy download en_core_web_sm
        ```

5.  **Train the Model:**
    *   Run the main training script. This will load data, preprocess, train using cross-validation, evaluate, and save `model.keras`, `label_encoder.pkl`, and `vectorizer.pkl`.
    ```bash
    python main.py
    ```
    *   *Note:* Training can take a significant amount of time depending on your hardware and the dataset size.

6.  **Run the Flask API Server:**
    *   Once the model is trained and artifacts are saved, start the Flask server.
    ```bash
    python app.py
    ```
    *   The API will be available at `http://127.0.0.1:5000` (or the port specified). Check the `/health` endpoint first.

7.  **Interactive Testing (Optional):**
    *   You can test the saved model directly from the command line:
    ```bash
    python test_model.py
    ```

## API Usage (`/predict`)

*   **Method:** `POST`
*   **URL:** `http://<your-server-address>:5000/predict`
*   **Content-Type:** `application/json`
*   **Request Body (JSON):**
    ```json
    {
      "symptoms": "fever, headache, cough", // User symptoms (string)
      "language": "en", // Optional: "en" or "ro" (defaults to "en")
      "follow_up": "" // Optional: User's answer ("yes" or "no", "da" or "nu") to a previous follow-up question
    }
    ```
*   **Response Body (JSON):**
    ```json
    {
      "diagnosis_message": "The predicted illness is: Flu with confidence 85.34%\n\nDISCLAIMER: ...", // Diagnosis + Disclaimer
      "follow_up_question": "Besides the symptoms you mentioned, do you have a high fever (e.g., above 101°F or 38.3°C)?" // Follow-up question, if applicable (empty string otherwise)
    }
    ```

## Dependencies

Key Python packages are listed in `requirements.txt`. Main dependencies include:

*   `tensorflow`: For building and training the neural network.
*   `scikit-learn`: For preprocessing (LabelEncoder), evaluation metrics, and model selection (KFold).
*   `pandas`: For data manipulation.
*   `numpy`: For numerical operations.
*   `spacy`: For advanced NLP preprocessing (lemmatization, etc.).
*   `nltk`: For NLP resources (WordNet for synonyms, stopwords).
*   `imbalanced-learn`: For SMOTE oversampling.
*   `Flask` & `Flask-Cors`: For creating the web API.
*   `joblib`: For saving/loading scikit-learn objects (encoder, vectorizer).

---

*(Optional: Add sections on Model Architecture details, Future Improvements, Contribution guidelines, etc.)*