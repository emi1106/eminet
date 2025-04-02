import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import re
import numpy as np
import random
from imblearn.over_sampling import SMOTE
import spacy

nltk.download('stopwords')
nltk.download('wordnet')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    # Use spaCy for advanced text processing
    doc = nlp(text)
    words = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(words)

def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stopwords.words('english')]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return new_words

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

def augment_text(text, num_augmented):
    words = text.split()
    augmented_texts = []
    for _ in range(num_augmented):
        new_words = synonym_replacement(words, n=1)
        augmented_texts.append(' '.join(new_words))
    return augmented_texts

def augment_data(data, num_augmented=5):
    augmented_data = []
    for index, row in data.iterrows():
        augmented_texts = augment_text(row['Symptoms'], num_augmented)
        for text in augmented_texts:
            augmented_data.append({'Symptoms': text, 'Illness': row['Illness']})
    return pd.DataFrame(augmented_data)

def handle_problematic_classes(data):
    """Remove or augment problematic classes based on their characteristics"""
    
    # Add flu-specific symptom variations
    flu_variations = [
        {'Illness': 'Flu', 'Symptoms': 'high fever, muscle aches, exhaustion, dry cough'},
        {'Illness': 'Flu', 'Symptoms': 'chills, body pain, fatigue, coughing, sore throat'},
        {'Illness': 'Flu', 'Symptoms': 'severe fatigue, fever, headache, congestion'},
        {'Illness': 'Flu', 'Symptoms': 'influenza symptoms, fever, weakness, respiratory symptoms'},
        {'Illness': 'Flu', 'Symptoms': 'high temperature, body aches, tiredness, chest discomfort'}
    ]
    data = pd.concat([data, pd.DataFrame(flu_variations)])
    
    # Classes to remove (those with consistently poor performance or no samples)
    remove_classes = [
        'Meningitis',       # Class 18
        'Hypercalcemia'     # Class 81
    ]
    
    # Classes to augment (those with potential for improvement)
    augment_classes = {
        'Flu': 15,                      # Class 6
        'Urinary Tract Infection': 10,  # Class 37
        'Acne': 8,                      # Class 54
        'Boils': 8,                     # Class 60
        'Hernia': 12                    # Class 80
    }
    
    # Remove problematic classes
    data = data[~data['Illness'].isin(remove_classes)]
    
    # Create extra augmented samples for classes that need improvement
    extra_samples = []
    for illness, num_aug in augment_classes.items():
        illness_data = data[data['Illness'] == illness]
        if not illness_data.empty:
            for _, row in illness_data.iterrows():
                augmented_texts = augment_text(row['Symptoms'], num_augmented=num_aug)
                for text in augmented_texts:
                    extra_samples.append({'Symptoms': text, 'Illness': illness})
    
    if extra_samples:
        data = pd.concat([data, pd.DataFrame(extra_samples)])
    
    return data

def load_and_preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Handle missing data
    data.dropna(subset=['Symptoms'], inplace=True)
    
    # Handle problematic classes
    data = handle_problematic_classes(data)
    
    # Encode categorical labels
    label_encoder = LabelEncoder()
    data['Illness'] = label_encoder.fit_transform(data['Illness'])
    
    # Augment data
    augmented_data = augment_data(data)
    data = pd.concat([data, augmented_data])
    
    # Split the data into features and target
    X = data['Symptoms']
    y = data['Illness']
    
    # Convert the symptoms to a suitable format using TF-IDF vectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Use bigrams
    X = vectorizer.fit_transform(X)
    
    # Convert the sparse matrix to a dense format
    X = X.toarray()
    
    # Simple SMOTE balancing
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)
    
    # Use standard train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Debugging: Print samples of the preprocessed data and labels
    print("Sample preprocessed data:", X_train[:5])
    print("Sample labels:", y_train[:5])
    
    return X_train, X_test, y_train, y_test, label_encoder, vectorizer
