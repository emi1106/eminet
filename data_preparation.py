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

nltk.download('stopwords')
nltk.download('wordnet')

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
    augmented_df = pd.DataFrame(augmented_data)
    return augmented_df

def load_and_preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Handle missing data
    data.dropna(subset=['Symptoms'], inplace=True)
    
    # Skip preprocessing symptoms
    # data['Symptoms'] = data['Symptoms'].apply(preprocess_text)
    
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
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)
    
    # Convert the sparse matrix to a dense format
    X = X.toarray()
    
    # Debugging: Print class distribution before splitting
    unique, counts = np.unique(y, return_counts=True)
    print("Class distribution before splitting:", dict(zip(unique, counts)))
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Debugging: Print samples of the preprocessed data and labels
    print("Sample preprocessed data:", X_train[:5])
    print("Sample labels:", y_train[:5])
    
    return X_train, X_test, y_train, y_test, label_encoder, vectorizer
