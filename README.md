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

5. **Run the main script:**
    ```sh
    python main.py
    ```

## File Descriptions

- `data_preparation.py`: Handles data loading and preprocessing.
- `model.py`: Defines and trains the neural network model.
- `evaluate.py`: Evaluates the trained model.
- `main.py`: Main entry point to run the entire process.

## Requirements

Make sure to have the following packages installed:
- pandas
- scikit-learn
- tensorflow

You can install these packages using the provided `requirements.txt` file.
```