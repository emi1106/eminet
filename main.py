import os
import numpy as np
import joblib
import logging
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf

# Import project modules
from data_preparation import load_and_preprocess_data
from model import build_model, train_model # Use the unified train_model
from evaluate import evaluate_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
DATA_FILE = 'illnesses_and_symptoms.csv'
N_SPLITS = 5  # Number of folds for cross-validation
EPOCHS = 100 # Reduced epochs, relying more on EarlyStopping
BATCH_SIZE = 32
TEST_SIZE = 0.2 # Hold-out test set size
RANDOM_STATE = 42

MODEL_SAVE_PATH = 'model.keras'
ENCODER_SAVE_PATH = 'label_encoder.pkl'
VECTORIZER_SAVE_PATH = 'vectorizer.pkl'

def main():
    """
    Main function to run the data preparation, model training (with CV),
    evaluation, and saving of artifacts.
    """
    # --- 1. Load and Preprocess Data ---
    # load_and_preprocess_data now returns the full processed dataset
    X_processed, y_processed, label_encoder, vectorizer = load_and_preprocess_data(DATA_FILE)

    if X_processed is None:
        logging.error("Data loading failed. Exiting.")
        return

    num_classes = len(label_encoder.classes_)
    logging.info(f"Data loaded. Features shape: {X_processed.shape}, Labels shape: {y_processed.shape}, Classes: {num_classes}")

    # --- 2. Create Train/Test Split ---
    # Split the *processed* data into training+validation and a final test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_processed, y_processed,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_processed # Stratify to maintain class distribution
    )
    logging.info(f"Train/Validation set size: {X_train_val.shape[0]}, Test set size: {X_test.shape[0]}")

    # --- 3. Cross-Validation Training ---
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_models = []
    fold_histories = []
    fold_accuracies = [] # Store validation accuracy for each fold

    logging.info(f"Starting {N_SPLITS}-Fold Cross-Validation...")

    for fold, (train_index, val_index) in enumerate(skf.split(X_train_val, y_train_val)):
        logging.info(f"\n--- Training Fold {fold + 1}/{N_SPLITS} ---")
        X_train, X_val = X_train_val[train_index], X_train_val[val_index]
        y_train, y_val = y_train_val[train_index], y_train_val[val_index]

        try:
            # Build a new model instance for each fold
            model = build_model(input_shape=X_train.shape[1], num_classes=num_classes)
            logging.info(f"Built model for fold {fold + 1}.")
            # model.summary(print_fn=logging.info) # Optional: log model summary

            # Train the model using the function from model.py
            history = train_model(
                model,
                X_train, y_train,
                X_val, y_val, # Pass validation data directly
                epochs=EPOCHS,
                batch_size=BATCH_SIZE
            )
            logging.info(f"Fold {fold + 1} training completed.")

            # Evaluate on validation set for this fold
            val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
            logging.info(f"Fold {fold + 1} Validation Performance: Loss={val_loss:.4f}, Accuracy={val_accuracy:.4f}")

            # Store results
            fold_models.append(model)
            fold_histories.append(history)
            fold_accuracies.append(val_accuracy)

        except Exception as e:
            logging.exception(f"Error during Fold {fold + 1} training: {e}")
            # Decide whether to continue or stop if a fold fails
            # continue

    # --- 4. Post-Cross-Validation Summary ---
    if not fold_models:
        logging.error("No models were successfully trained during cross-validation. Exiting.")
        return

    mean_cv_accuracy = np.mean(fold_accuracies)
    std_cv_accuracy = np.std(fold_accuracies)
    best_fold_index = np.argmax(fold_accuracies)
    best_model = fold_models[best_fold_index]

    logging.info("\n--- Cross-Validation Summary ---")
    logging.info(f"Mean Validation Accuracy: {mean_cv_accuracy:.4f}")
    logging.info(f"Std Dev Validation Accuracy: {std_cv_accuracy:.4f}")
    logging.info(f"Best Fold: {best_fold_index + 1} with Accuracy: {fold_accuracies[best_fold_index]:.4f}")

    # --- 5. Final Evaluation on Hold-Out Test Set ---
    logging.info("\n--- Final Evaluation on Hold-Out Test Set ---")
    logging.info("Evaluating the best model from cross-validation...")
    evaluate_model(
        best_model,
        X_test, y_test,
        label_encoder=label_encoder,
        # Optionally pass train data from the best fold for overfitting check comparison
        X_train=X_train_val[skf.split(X_train_val, y_train_val)[best_fold_index][0]],
        y_train=y_train_val[skf.split(X_train_val, y_train_val)[best_fold_index][0]],
        is_keras_model=True
    )

    # --- 6. Save the Best Model and Artifacts ---
    logging.info(f"\nSaving the best model (from fold {best_fold_index + 1}) and preprocessing tools...")
    try:
        best_model.save(MODEL_SAVE_PATH)
        joblib.dump(label_encoder, ENCODER_SAVE_PATH)
        joblib.dump(vectorizer, VECTORIZER_SAVE_PATH)
        logging.info(f"Model saved to {MODEL_SAVE_PATH}")
        logging.info(f"Label Encoder saved to {ENCODER_SAVE_PATH}")
        logging.info(f"Vectorizer saved to {VECTORIZER_SAVE_PATH}")
    except Exception as e:
        logging.exception(f"Error saving model or artifacts: {e}")

    logging.info("\n--- Main script finished ---")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)
    random.seed(RANDOM_STATE) # For python's random module used in augmentation

    # Ensure NLTK data is available (user responsibility - see README)
    try:
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/stopwords')
    except LookupError as e:
        logging.error(f"NLTK data missing: {e}. Please download using nltk.download('wordnet') and nltk.download('stopwords') or follow README setup.")
        exit()

    main()