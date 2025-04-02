import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.regularizers import l1_l2
import numpy as np

def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=(input_shape,)))
    
    # First layer with moderate width
    model.add(Dense(256, activation='relu', 
                   kernel_regularizer=l1_l2(l1=0.005, l2=0.005)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # Hidden layers with skip connections
    for units in [128, 64]:
        x = model.layers[-1].output
        x = Dense(units, activation='relu',
                 kernel_regularizer=l1_l2(l1=0.005, l2=0.005))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        model.add(Dense(units, activation='relu'))
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0005,
        beta_1=0.9,
        beta_2=0.999
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model(model, X_train, y_train, epochs=150, batch_size=32):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            min_delta=0.0005
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.00001
        )
    ]
    
    # Custom callback to show metrics and suppress the default progress bar
    class CustomCallback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            print(f"\nEpoch {epoch+1}/{epochs}")
        
        def on_epoch_end(self, epoch, logs=None):
            # Update the visualizer
            update_visualizer(epoch, logs)
            # Print metrics
            train_loss = logs.get('loss', 0)
            train_acc = logs.get('accuracy', 0)
            val_loss = logs.get('val_loss', 0)
            val_acc = logs.get('val_accuracy', 0)
            print(f"loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")
    
    # Add the custom callback to suppress progress bar and show metrics
    callbacks.append(CustomCallback())

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=callbacks,
        shuffle=True,
    )
    return history

def compute_class_weights(y):
    class_counts = np.bincount(y)
    total = len(y)
    
    # More aggressive weighting for problematic classes
    weights = {}
    median_count = np.median(class_counts)
    
    for i, count in enumerate(class_counts):
        if count == 0:
            weights[i] = 5.0  # High weight for classes with no samples
        elif i in [6, 41, 71, 84]:  # Problematic classes
            weights[i] = 5.0
        elif count < median_count / 2:
            weights[i] = (total / (len(class_counts) * count)) ** 1.5
        else:
            weights[i] = total / (len(class_counts) * count)
    
    return weights
