"""
Translation module for multilingual support in the medical diagnosis app.
Supports English and Romanian languages.
"""

# Common medical terms translations (English to Romanian)
MEDICAL_TERMS = {
    # Symptoms translations
    "headache": "durere de cap",
    "fever": "febra",
    "cough": "tuse",
    "fatigue": "oboseala",
    "nausea": "greata",
    "vomiting": "varsaturi,vomitat",
    "diarrhea": "diaree",
    "abdominal pain": "durere abdominala",
    "chest pain": "durere in piept",
    "shortness of breath": "dificultate in respiratie",
    "sore throat": "durere in gat",
    "runny nose": "nas care curge",
    "muscle ache": "durere musculara",
    "joint pain": "durere articulara",
    "rash": "eruptie cutanata",
    "dizziness": "ameteala",
    "chills": "frisoane",
    "high blood pressure": "tensiune arteriala ridicata",
    "low blood pressure": "tensiune arteriala scazuta",
    
    # Common illnesses translations
    "common cold": "raceala",
    "flu": "gripa",
    "covid-19": "covid-19",
    "pneumonia": "pneumonie",
    "bronchitis": "bronsita",
    "asthma": "astm",
    "allergies": "alergii",
    "sinusitis": "sinuzita",
    "gastroenteritis": "gastroenterita",
    "urinary tract infection": "infectie urinara",
    "migraine": "migrena",
    "hypertension": "hipertensiune",
    "diabetes": "diabet",
    "heart disease": "boala de inima",
    "stroke": "accident vascular cerebral",
    "cancer": "cancer",
    "depression": "depresie",
    "anxiety": "anxietate",
}

# Diagnosis message templates
DIAGNOSIS_TEMPLATES = {
    "en": {
        "prediction": "The predicted illness is: {illness} with confidence {confidence:.2%}",
        "more_likely": "The diagnosis of {illness} is more likely.",
        "less_likely": "The diagnosis of {illness} is less likely. Please consult a doctor for a more accurate diagnosis.",
        "emergency": "EMERGENCY: This could indicate a serious condition. Seek immediate medical attention.",
        "consult": "Please consult with a healthcare professional for proper diagnosis and treatment.",
    },
    "ro": {
        "prediction": "Boala prezisa este: {illness} cu incredere {confidence:.2%}",
        "more_likely": "Diagnosticul de {illness} este mai probabil.",
        "less_likely": "Diagnosticul de {illness} este mai putin probabil. Va rugam sa consultati un medic pentru un diagnostic mai precis.",
        "emergency": "URGENTA: Aceasta ar putea indica o afectiune grava. Cautati asistenta medicala imediata.",
        "consult": "Va rugam sa consultati un profesionist medical pentru diagnostic si tratament adecvat.",
    }
}

def get_diagnosis_message(language, template_key, **kwargs):
    """Get a translated diagnosis message with placeholders filled"""
    lang = language if language in DIAGNOSIS_TEMPLATES else "en"
    template = DIAGNOSIS_TEMPLATES[lang].get(template_key, DIAGNOSIS_TEMPLATES["en"][template_key])
    return template.format(**kwargs)

def translate_medical_term(term, to_romanian=False):
    """Translate a medical term between English and Romanian"""
    term = term.lower()
    if to_romanian:
        # English to Romanian
        return MEDICAL_TERMS.get(term, term)
    else:
        # Romanian to English
        for en_term, ro_term in MEDICAL_TERMS.items():
            # Handle multiple translations separated by comma
            ro_variations = ro_term.split(',')
            for variation in ro_variations:
                if term == variation.lower():
                    return en_term
        return term

def translate_illness(illness, language):
    """Translate an illness name to the target language"""
    if language == "ro":
        return translate_medical_term(illness, to_romanian=True)
    return illness  # Default to original (assumed English)
