import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

def load_and_preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Convert symptoms to lowercase
    data['Symptoms'] = data['Symptoms'].str.lower()
    
    # Encode categorical labels
    label_encoder = LabelEncoder()
    data['Illness'] = label_encoder.fit_transform(data['Illness'])
    
    # Split the data into features and target
    X = data['Symptoms']
    y = data['Illness']
    
    # Convert the symptoms to a suitable format using TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)
    
    # Convert the sparse matrix to a dense format
    X = X.toarray()
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, label_encoder, vectorizer
