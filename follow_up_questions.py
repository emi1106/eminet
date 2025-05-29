"""
This module contains follow-up questions for specific illnesses and symptoms,
supporting both English and Romanian.
"""

# Follow-up questions triggered by the *predicted illness*
# Format: "illness_name_lowercase": "Question text"
follow_up_by_illness = {
    "en": {
        "common cold": "Do you also have a runny or stuffy nose?",
        "flu": "Besides the symptoms you mentioned, do you have a high fever (e.g., above 101°F or 38.3°C)?",
        "covid-19": "Have you experienced a recent loss of taste or smell?",
        "migraine": "Are your headaches typically accompanied by sensitivity to light or sound?",
        "allergies": "Do your symptoms seem to worsen in specific environments (like outdoors) or during certain seasons?",
        "asthma": "Is there wheezing or a whistling sound when you breathe?",
        "bronchitis": "Have you been coughing up thick or colored mucus (yellow/green)?",
        "pneumonia": "Do you feel sharp chest pain, especially when breathing deeply or coughing?",
        "sinusitis": "Do you feel pressure or tenderness around your eyes, cheeks, or forehead?",
        "urinary tract infection": "Do you experience a burning sensation when you urinate?",
        "gastroenteritis": "Besides nausea/vomiting/diarrhea, do you have significant stomach cramps?",
        "hypertension": "Have you had your blood pressure checked recently? Was it high?", # Less direct symptom question
        "diabetes": "Have you noticed excessive thirst or needing to urinate much more often than usual?",
        "heart disease": "Do you get short of breath easily during physical activity?",
        "heart attack": "Are you experiencing chest pain or discomfort, possibly like pressure or squeezing, which might radiate to your arm, neck, or jaw? (If so, please seek immediate medical attention!)",
        "anemia": "Do you feel unusually tired, weak, or look pale?",
        "appendicitis": "Is the abdominal pain mostly concentrated in the lower right part of your belly?",
        "gerd": "Do you often experience heartburn or acid reflux, especially after eating or when lying down?",
        "ibs": "Does your abdominal discomfort seem to change (improve or worsen) after a bowel movement?",
    },
    "ro": {
        "common cold": "Aveți și nasul înfundat sau vă curge nasul?",
        "flu": "Pe lângă simptomele menționate, aveți febră mare (ex. peste 38.3°C)?",
        "covid-19": "Ați observat recent o pierdere a gustului sau mirosului?",
        "migraine": "Durerile de cap sunt de obicei însoțite de sensibilitate la lumină sau sunet?",
        "allergies": "Simptomele par să se agraveze în anumite medii (precum afară) sau în anumite anotimpuri?",
        "asthma": "Se aude un șuierat când respirați?",
        "bronchitis": "Tușiți mucus gros sau colorat (galben/verde)?",
        "pneumonia": "Simțiți o durere ascuțită în piept, mai ales când respirați adânc sau tușiți?",
        "sinusitis": "Simțiți presiune sau sensibilitate în jurul ochilor, obrajilor sau pe frunte?",
        "urinary tract infection": "Simțiți o senzație de arsură când urinați?",
        "gastroenteritis": "Pe lângă greață/vărsături/diaree, aveți crampe stomacale semnificative?",
        "hypertension": "V-ați verificat recent tensiunea arterială? A fost ridicată?",
        "diabetes": "Ați observat o sete excesivă sau nevoia de a urina mult mai des decât de obicei?",
        "heart disease": "Obosiți ușor sau aveți dificultăți de respirație în timpul activității fizice?",
        "heart attack": "Simțiți durere sau disconfort în piept, posibil ca o presiune sau strângere, care ar putea radia către braț, gât sau maxilar? (Dacă da, vă rugăm să căutați imediat asistență medicală!)",
        "anemia": "Vă simțiți neobișnuit de obosit, slăbit sau sunteți palid?",
        "appendicitis": "Durerea abdominală este concentrată mai ales în partea dreaptă jos a abdomenului?",
        "gerd": "Aveți des arsuri la stomac sau reflux acid, mai ales după masă sau când stați întins?",
        "ibs": "Disconfortul abdominal pare să se modifice (ameliorare sau agravare) după defecație?",
    }
}

# Follow-up questions triggered by specific *keywords found in symptoms*
# Format: "symptom_keyword": [("potential_condition_en", "Question text"), ...]
# These are asked if no illness-specific question is triggered.
detailed_follow_up_by_symptom = {
    "en": {
        "headache": [
            ("migraine", "Is the headache pulsating and mainly on one side?"),
            ("tension headache", "Does the headache feel like a tight band around your head?"),
        ],
        "chest pain": [
            ("heart attack", "Does the pain feel like pressure or squeezing and radiate to your arm, neck, or jaw? (Seek immediate help if yes!)"),
            ("gerd", "Is the chest pain a burning sensation, possibly worse after eating?"),
        ],
        "fatigue": [
            ("anemia", "Besides tiredness, do you feel weak, dizzy, or look pale?"),
            ("depression", "Along with fatigue, have you felt persistently sad or lost interest in things you usually enjoy?"),
        ],
        "abdominal pain": [
             ("appendicitis", "Is the pain severe and located primarily in the lower right abdomen?"),
             ("gallstones", "Does the pain often occur after eating, especially fatty meals?"),
        ],
        "cough": [
            ("bronchitis", "Is the cough producing yellow or green phlegm?"),
            ("asthma", "Does the cough come with wheezing or shortness of breath?"),
        ],
        "breath": [ # Catches "shortness of breath", "difficulty breathing" etc.
            ("asthma", "Does the shortness of breath come in episodes, possibly triggered by exercise or allergens?"),
            ("heart failure", "Is the shortness of breath worse when lying down, or do you have swelling in your legs?"),
        ],
        "dizzy": [ # Catches "dizziness", "dizzy"
            ("low blood pressure", "Do you feel dizzy mostly when standing up quickly?"),
            ("inner ear problems", "Is the dizziness more like a spinning sensation (vertigo)?"),
        ]
    },
    "ro": {
         "headache": [ # Matches 'durere cap' after preprocessing
            ("migraine", "Durerea este pulsantă și predominant pe o singură parte a capului?"),
            ("tension headache", "Simțiți durerea ca o bandă strânsă în jurul capului?"),
        ],
        "chest pain": [ # Matches 'durere piept'
            ("heart attack", "Simțiți durerea ca o presiune sau strângere, radiind spre braț, gât sau maxilar? (Căutați ajutor imediat dacă da!)"),
            ("gerd", "Durerea în piept este ca o arsură, posibil mai rea după masă?"),
        ],
        "fatigue": [ # Matches 'oboseala'
            ("anemia", "Pe lângă oboseală, vă simțiți slăbit, amețit sau păreți palid?"),
            ("depression", "Împreună cu oboseala, v-ați simțit persistent trist sau ați pierdut interesul pentru lucrurile care vă plac de obicei?"),
        ],
        "abdominal pain": [ # Matches 'durere abdominala'
             ("appendicitis", "Durerea este severă și localizată predominant în partea dreaptă jos a abdomenului?"),
             ("gallstones", "Durerea apare des după masă, în special după cele grase?"),
        ],
        "cough": [ # Matches 'tuse'
            ("bronchitis", "Tușiți flegmă galbenă sau verde?"),
            ("asthma", "Tusea este însoțită de respirație șuierătoare sau dificultăți de respirație?"),
        ],
        "breath": [ # Matches 'respiratie', 'dificultate' etc.
            ("asthma", "Dificultatea de respirație apare în episoade, posibil declanșată de efort sau alergeni?"),
            ("heart failure", "Dificultatea de respirație este mai rea când stați întins sau aveți picioarele umflate?"),
        ],
        "dizzy": [ # Matches 'ameteala'
            ("low blood pressure", "Amețiți mai ales când vă ridicați repede în picioare?"),
            ("inner ear problems", "Amețeala seamănă mai mult cu o senzație de rotire (vertij)?"),
        ]
    }
}


def get_follow_up_question(illness_en_lowercase, language="en"):
    """
    Returns an appropriate follow-up question based on the predicted illness
    and requested language. Returns an empty string if no specific question exists.
    """
    lang_dict = follow_up_by_illness.get(language, follow_up_by_illness["en"])
    return lang_dict.get(illness_en_lowercase, "")

def get_detailed_follow_up(symptom_keyword, language="en"):
    """
    Returns a list of potential detailed follow-up questions [(condition_en, question_text), ...]
    if the symptom keyword matches. Returns an empty list otherwise.
    """
    lang_dict = detailed_follow_up_by_symptom.get(language, detailed_follow_up_by_symptom["en"])
    return lang_dict.get(symptom_keyword, [])