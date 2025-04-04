"""
This module contains follow-up questions for specific illnesses and symptoms.
"""

# Follow-up questions for specific illnesses
follow_up_questions = {
    "common cold": "Do you have a runny or stuffy nose?",
    "flu": "Do you have a high fever above 101°F (38.3°C)?",
    "covid-19": "Have you lost your sense of taste or smell?",
    "migraine": "Do you experience sensitivity to light or sound during headaches?",
    "allergies": "Do your symptoms worsen in specific environments or seasons?",
    "asthma": "Do you have a family history of asthma?",
    "bronchitis": "Have you been coughing up colored mucus?",
    "pneumonia": "Do you have chest pain that worsens when you breathe deeply?",
    "sinusitis": "Do you feel pressure or pain around your eyes or forehead?",
    "urinary tract infection": "Do you feel a burning sensation when urinating?",
    "gastroenteritis": "Have you been in contact with anyone with similar symptoms?",
    "hypertension": "Do you have a family history of high blood pressure?",
    "diabetes": "Do you experience excessive thirst and frequent urination?",
    "heart disease": "Do you experience shortness of breath during physical activity?",
    "anemia": "Do you feel unusually tired or weak?",
    "appendicitis": "Is the pain located in the lower right side of your abdomen?",
    "gerd": "Do your symptoms worsen after eating or when lying down?",
    "ibs": "Does your abdominal pain improve after bowel movements?",
}

# Detailed follow-up questions based on specific symptoms
detailed_follow_up = {
    "headache": [
        ("migraine", "Is your headache accompanied by visual disturbances like flashing lights?"),
        ("tension headache", "Is your headache accompanied by a feeling of pressure around your head?"),
        ("cluster headache", "Is the pain severe and concentrated around one eye?")
    ],
    "chest pain": [
        ("heart attack", "Is the pain radiating to your arm, jaw, or back?"),
        ("angina", "Does the pain occur during physical activity and subside with rest?"),
        ("pneumonia", "Is the pain worse when you take a deep breath?")
    ],
    "fatigue": [
        ("anemia", "Have you noticed paleness of skin or brittle nails?"),
        ("depression", "Have you lost interest in activities you used to enjoy?"),
        ("hypothyroidism", "Have you gained weight unexpectedly?"),
        ("chronic fatigue syndrome", "Has your fatigue lasted for more than 6 months?")
    ],
    "abdominal pain": [
        ("appendicitis", "Is the pain most intense in the lower right side of your abdomen?"),
        ("gallstones", "Does the pain occur after eating fatty foods?"),
        ("ulcer", "Does eating make the pain better or worse?"),
        ("ibs", "Is the pain relieved by having a bowel movement?")
    ],
    "cough": [
        ("bronchitis", "Are you coughing up yellow or green mucus?"),
        ("asthma", "Do you also experience wheezing or shortness of breath?"),
        ("covid-19", "Have you lost your sense of taste or smell recently?"),
        ("pneumonia", "Do you have a high fever and chest pain?")
    ],
    "shortness of breath": [
        ("asthma", "Does it come in episodes and get worse with exercise or allergic triggers?"),
        ("heart failure", "Do you notice swelling in your legs or feet?"),
        ("covid-19", "Have you been in contact with anyone who tested positive for COVID-19?"),
        ("pulmonary embolism", "Did the shortness of breath start suddenly?")
    ],
    "dizziness": [
        ("inner ear problems", "Do you also experience ringing in your ears or hearing loss?"),
        ("low blood pressure", "Does the dizziness occur when standing up quickly?"),
        ("anxiety", "Do you also experience rapid heartbeat or feelings of panic?"),
        ("stroke", "Do you have other symptoms like slurred speech or facial drooping?")
    ]
}

# Romanian translations of follow-up questions could be added here
# This would be implemented in a more comprehensive solution
