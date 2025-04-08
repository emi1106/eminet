import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
# Note: NLTK data download should be done manually once after installation
# See README.md
from nltk.corpus import stopwords, wordnet
# from nltk.stem import WordNetLemmatizer # spaCy handles lemmatization
import re
import numpy as np
import random
from imblearn.over_sampling import SMOTE
import spacy
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load spaCy model (consider error handling if spaCy model not found)
try:
    nlp = spacy.load("en_core_web_sm")
    logging.info("spaCy model 'en_core_web_sm' loaded successfully.")
except OSError:
    logging.error("spaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'")
    # Depending on requirements, you might exit or fallback to simpler processing
    # For this script, we assume spaCy is available as it's crucial for preprocessing.
    exit()

def preprocess_text(text):
    """
    Preprocesses text using spaCy: lowercasing, lemmatization, removing stopwords and non-alpha tokens.
    """
    if not isinstance(text, str):
        text = str(text) # Ensure text is string

    doc = nlp(text.lower())
    words = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(words)

def get_synonyms(word):
    """Gets synonyms for a word using WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            # Simple check to avoid obscure synonyms (optional)
            if '_' not in lemma.name():
                synonyms.add(lemma.name())
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

def synonym_replacement(words, n):
    """Replaces n words in a list with their synonyms."""
    new_words = words.copy()
    # Use a predefined list of stopwords if NLTK download isn't guaranteed
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        logging.warning("NLTK stopwords not found. Using a basic list. Run nltk.download('stopwords')")
        stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'}

    random_word_list = list(set([word for word in words if word not in stop_words and len(word) > 2])) # Filter short words too
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms).replace('_', ' ') # Handle multi-word synonyms
            new_words = [synonym if word == random_word else word for word in new_words]
            # Re-split in case synonym was multi-word
            new_words = ' '.join(new_words).split()
            num_replaced += 1
        if num_replaced >= n:
            break
    return ' '.join(new_words)

def augment_text_synonyms(text, num_augmented=1):
    """Augments text by replacing words with synonyms."""
    words = text.split()
    augmented_texts = []
    for _ in range(num_augmented):
        # Replace roughly 10% of non-stop words, minimum 1
        n_replace = max(1, int(len(words) * 0.1))
        augmented_sentence = synonym_replacement(words, n=n_replace)
        augmented_texts.append(augmented_sentence)
    return augmented_texts

def handle_problematic_classes(data):
    """
    Applies specific augmentation or removal strategies for classes
    known to perform poorly or have limited data.
    """
    logging.info("Handling specific classes...")

    # Example: Add specific variations for Flu
    flu_variations = [
        {'Illness': 'Flu', 'Symptoms': 'high fever, muscle aches, exhaustion, dry cough'},
        {'Illness': 'Flu', 'Symptoms': 'chills, body pain, fatigue, coughing, sore throat'},
        {'Illness': 'Flu', 'Symptoms': 'severe fatigue, fever, headache, congestion'},
        {'Illness': 'Flu', 'Symptoms': 'influenza symptoms, fever, weakness, respiratory symptoms'},
        {'Illness': 'Flu', 'Symptoms': 'high temperature, body aches, tiredness, chest discomfort'}
    ]
    if 'Flu' in data['Illness'].unique():
        data = pd.concat([data, pd.DataFrame(flu_variations)], ignore_index=True)
        logging.info(f"Added {len(flu_variations)} specific variations for 'Flu'.")

    # Example: Classes to remove
    remove_classes = ['Meningitis', 'Hypercalcemia'] # Based on prior analysis (adjust as needed)
    original_count = len(data)
    data = data[~data['Illness'].isin(remove_classes)]
    if len(data) < original_count:
        logging.info(f"Removed classes: {remove_classes}. {original_count - len(data)} rows removed.")

    # Example: Classes needing extra augmentation
    augment_classes_extra = {
        'Flu': 10, # Add 10 extra synonym-augmented samples for each existing Flu sample
        'Urinary Tract Infection': 8,
        # Add others as identified
    }
    extra_samples = []
    for illness, num_aug in augment_classes_extra.items():
        illness_data = data[data['Illness'] == illness]
        if not illness_data.empty:
            logging.info(f"Generating {num_aug} extra augmented samples per instance for '{illness}'.")
            for _, row in illness_data.iterrows():
                # Preprocess *before* augmentation ensures cleaner input
                preprocessed_symptoms = preprocess_text(row['Symptoms'])
                if preprocessed_symptoms: # Ensure not empty after preprocessing
                    augmented_texts = augment_text_synonyms(preprocessed_symptoms, num_augmented=num_aug)
                    for text in augmented_texts:
                        extra_samples.append({'Symptoms': text, 'Illness': illness})

    if extra_samples:
        data = pd.concat([data, pd.DataFrame(extra_samples)], ignore_index=True)
        logging.info(f"Added {len(extra_samples)} extra augmented samples.")

    return data

def load_and_preprocess_data(file_path, augment=True, use_smote=True):
    """
    Loads data, preprocesses text, handles specific classes, optionally augments
    data using synonym replacement, and optionally applies SMOTE.

    Args:
        file_path (str): Path to the CSV file.
        augment (bool): Whether to apply synonym augmentation. Default True.
        use_smote (bool): Whether to apply SMOTE oversampling. Default True.

    Returns:
        tuple: (X_processed, y_processed, label_encoder, vectorizer)
               - X_processed: Processed feature matrix (TF-IDF).
               - y_processed: Processed labels (encoded).
               - label_encoder: Fitted LabelEncoder.
               - vectorizer: Fitted TfidfVectorizer.
    """
    logging.info(f"Loading data from {file_path}...")
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        logging.error(f"Error: File not found at {file_path}")
        return None, None, None, None

    logging.info(f"Initial dataset shape: {data.shape}")

    # Basic cleaning
    data.dropna(subset=['Symptoms'], inplace=True)
    data.drop_duplicates(subset=['Symptoms', 'Illness'], inplace=True)
    logging.info(f"Shape after dropping NA/duplicates: {data.shape}")

    # Handle specific problematic classes
    data = handle_problematic_classes(data)
    logging.info(f"Shape after handling specific classes: {data.shape}")

    # Preprocess text symptoms
    logging.info("Preprocessing text symptoms using spaCy...")
    data['Processed_Symptoms'] = data['Symptoms'].apply(preprocess_text)

    # Filter out rows where preprocessing resulted in empty strings
    original_count = len(data)
    data = data[data['Processed_Symptoms'].str.len() > 0]
    if len(data) < original_count:
        logging.warning(f"Removed {original_count - len(data)} rows with empty symptoms after preprocessing.")

    # Optional: Augment data using synonym replacement on preprocessed text
    if augment:
        logging.info("Augmenting data using synonym replacement...")
        augmented_data = []
        for index, row in data.iterrows():
            # Augment the *preprocessed* text
            augmented_texts = augment_text_synonyms(row['Processed_Symptoms'], num_augmented=2) # Augment each sample twice
            for text in augmented_texts:
                 if text.strip(): # Ensure augmented text is not empty
                    augmented_data.append({'Processed_Symptoms': text, 'Illness': row['Illness']})

        if augmented_data:
            data = pd.concat([data, pd.DataFrame(augmented_data)], ignore_index=True)
            logging.info(f"Added {len(augmented_data)} augmented samples. New shape: {data.shape}")

    # Encode labels BEFORE potential SMOTE
    label_encoder = LabelEncoder()
    data['Illness_Encoded'] = label_encoder.fit_transform(data['Illness'])
    logging.info(f"Encoded {len(label_encoder.classes_)} unique classes.")

    # Prepare features (X) and target (y)
    X = data['Processed_Symptoms']
    y = data['Illness_Encoded'].values # Use .values for numpy array

    # TF-IDF Vectorization
    logging.info("Applying TF-IDF Vectorization...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000) # Limit features
    X_tfidf = vectorizer.fit_transform(X).toarray()
    logging.info(f"TF-IDF matrix shape: {X_tfidf.shape}")

    # Optional: Apply SMOTE for balancing
    if use_smote:
        logging.info("Applying SMOTE for class balancing...")
        # Ensure y is 1D array
        if y.ndim > 1:
            y = y.ravel()
        smote = SMOTE(random_state=42, k_neighbors=max(1, min(5, np.min(np.bincount(y))-1 if np.min(np.bincount(y)) > 1 else 1))) # Adjust k_neighbors safely
        try:
            X_resampled, y_resampled = smote.fit_resample(X_tfidf, y)
            logging.info(f"Shape after SMOTE: {X_resampled.shape}, {y_resampled.shape}")
            X_processed = X_resampled
            y_processed = y_resampled
        except ValueError as e:
            logging.error(f"SMOTE failed: {e}. This might happen if a class has too few samples even after augmentation. Proceeding without SMOTE.")
            X_processed = X_tfidf
            y_processed = y
    else:
        X_processed = X_tfidf
        y_processed = y

    logging.info("Data preparation complete.")
    return X_processed, y_processed, label_encoder, vectorizer

# Example usage (if run directly)
if __name__ == "__main__":
    # Ensure NLTK data is downloaded (run these lines once manually if needed)
    # nltk.download('stopwords')
    # nltk.download('wordnet')

    # Test the function
    file = 'illnesses_and_symptoms.csv' # Assuming CSV is in the same directory
    X_proc, y_proc, le, vec = load_and_preprocess_data(file)

    if X_proc is not None:
        print("\n--- Data Loading Summary ---")
        print(f"Processed features shape: {X_proc.shape}")
        print(f"Processed labels shape: {y_proc.shape}")
        print(f"Number of unique classes: {len(le.classes_)}")
        print(f"Vectorizer features: {len(vec.get_feature_names_out())}")
        print("\nSample processed labels:", y_proc[:10])
        print("\nClass distribution after processing:")
        unique, counts = np.unique(y_proc, return_counts=True)
        print(dict(zip(le.inverse_transform(unique), counts)))