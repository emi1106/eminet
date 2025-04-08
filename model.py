import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import logging
import numpy as np # Keep numpy import

def build_model(input_shape, num_classes):
    """Builds the Keras Sequential model."""
    logging.info(f"Building model for {num_classes} classes with input shape {input_shape}.")
    model = Sequential(name="Symptom_Diagnosis_NN")
    model.add(Input(shape=(input_shape,), name="Input_Layer"))

    # Input Layer -> Dense + Regularization + BN + Dropout
    model.add(Dense(256, activation='relu',
                    kernel_regularizer=l1_l2(l1=0.001, l2=0.001), # Slightly reduced regularization
                    name="Dense_1"))
    model.add(BatchNormalization(name="BatchNorm_1"))
    model.add(Dropout(0.3, name="Dropout_1")) # Slightly increased dropout

    # Hidden Layer 1
    model.add(Dense(128, activation='relu',
                    kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                    name="Dense_2"))
    model.add(BatchNormalization(name="BatchNorm_2"))
    model.add(Dropout(0.4, name="Dropout_2")) # Increased dropout

    # Hidden Layer 2
    model.add(Dense(64, activation='relu',
                   kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                   name="Dense_3"))
    model.add(BatchNormalization(name="BatchNorm_3"))
    model.add(Dropout(0.4, name="Dropout_3"))

    # Output Layer
    model.add(Dense(num_classes, activation='softmax', name="Output_Layer"))

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # Standard Adam LR

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy', # Suitable for integer labels
        metrics=['accuracy']
    )
    logging.info("Model compiled successfully.")
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """
    Trains the Keras model with Early Stopping and Learning Rate Reduction.

    Args:
        model: Compiled Keras model.
        X_train, y_train: Training data and labels.
        X_val, y_val: Validation data and labels.
        epochs (int): Maximum number of epochs.
        batch_size (int): Batch size for training.

    Returns:
        History object from model.fit().
    """
    logging.info(f"Starting training for max {epochs} epochs with batch size {batch_size}.")

    callbacks = [
        EarlyStopping(
            monitor='val_loss',   # Monitor validation loss
            patience=15,          # Stop after 15 epochs with no improvement
            restore_best_weights=True, # Restore weights from the best epoch
            min_delta=0.0005,     # Minimum change to qualify as improvement
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',   # Monitor validation loss
            factor=0.2,           # Reduce LR by a factor of 0.2
            patience=7,           # Reduce LR after 7 epochs with no improvement
            min_lr=1e-6,          # Minimum learning rate
            verbose=1
        )
        # Removed the CustomCallback as it wasn't fully implemented
        # and verbose=1 provides progress.
    ]

    # Ensure labels are numpy arrays
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val), # Provide validation data here
        callbacks=callbacks,
        shuffle=True,
        verbose=1 # Use Keras default progress bar (0=silent, 1=bar, 2=line per epoch)
    )
    logging.info("Training finished.")
    return history

# Removed compute_class_weights as SMOTE is handling imbalance in data_preparation.