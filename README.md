# Symptom-Based Diagnosis Neural Network

This project builds a neural network to diagnose based on input symptoms.

## Setup Instructions

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/eminet.git
    cd eminet
    ```

2. **Create a virtual environment:**
    ```sh
    python -m venv venv
    ```

3. **Activate the virtual environment:**
    - On Windows:
        ```sh
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```sh
        source venv/bin/activate
        ```

4. **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

5. **Download the spaCy language model:**
    ```sh
    python -m spacy download en_core_web_sm
    ```

6. **Train the model first:**
    ```sh
    python main.py
    ```

7. **Run the Flask web server:**
    ```sh
    python app.py
    ```

8. **For the Flutter App:**
   - Navigate to the Flutter app directory:
     ```sh
     cd flutter_app/frontend
     ```
   - Install dependencies:
     ```sh
     flutter pub get
     ```
   - Run the app:
     ```sh
     flutter run
     ```

## File Descriptions

- `data_preparation.py`: Handles data loading and preprocessing.
- `model.py`: Defines and trains the neural network model.
- `evaluate.py`: Evaluates the trained model.
- `main.py`: Main entry point to run the entire process.
- `app.py`: Flask server that provides an API for the model.
- `flutter_app/frontend`: Flutter mobile application that consumes the API.

## Requirements

The project requires the following packages:
- pandas, scikit-learn, tensorflow (for ML)
- flask, flask-cors (for the API)
- spaCy with English language model (for NLP)
- Flutter (for the mobile app)

You can install Python packages using the provided `requirements.txt` file.