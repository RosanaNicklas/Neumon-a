# Import all required libraries
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import seaborn as sns
import os
import datetime

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 15
LEARNING_RATE_PHASE1 = 1e-4
LEARNING_RATE_PHASE2 = 1e-5

def prepare_data(train_dir, val_dir, test_dir):
    """Prepare data generators and calculate class weights."""
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    # Validation and test generators
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    # Create data generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True)

    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False)

    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False)

    # Calculate class weights for imbalanced data
    classes = np.unique(train_generator.classes)
    weights = class_weight.compute_class_weight(
        'balanced',
        classes=classes,
        y=train_generator.classes)
    class_weights = dict(enumerate(weights))

    return train_generator, val_generator, test_generator, class_weights

def build_model():
    """Build and compile the EfficientNetV2 model."""
    # Load pre-trained base model
    base_model = EfficientNetV2B0(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3))
    base_model.trainable = False

    # Create custom model architecture
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE_PHASE1),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )

    return model

def train_model(model, train_gen, val_gen, class_weights):
    """Train the model in two phases."""
    # Callbacks for training
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True)
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=1e-7)

    # Phase 1: Train top layers
    print("\n=== Phase 1: Training Top Layers ===")
    history1 = model.fit(
        train_gen,
        epochs=EPOCHS_PHASE1,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=[early_stopping, reduce_lr]
    )

    # Phase 2: Fine-tune
    print("\n=== Phase 2: Fine-Tuning ===")
    model.layers[1].trainable = True  # Unfreeze base model
    
    # Only unfreeze last 30 layers
    for layer in model.layers[1].layers[:-30]:
        layer.trainable = False
    
    from keras.metrics import AUC, Precision, Recall
    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE_PHASE2),
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc'), Precision(name='precision'), Recall(name='recall')]

    )

    history2 = model.fit(
        train_gen,
        epochs=EPOCHS_PHASE2,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=[early_stopping, reduce_lr]
    )

    return model, history1, history2

def evaluate_model(model, test_gen):
    """Evaluate model performance on test set."""
    # Get evaluation metrics
    results = model.evaluate(test_gen)
    metrics = dict(zip(model.metrics_names, results))
    
    # Generate predictions
    y_pred = (model.predict(test_gen) > 0.5).astype(int)
    
    # Confusion matrix
    cm = confusion_matrix(test_gen.classes, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['NORMAL', 'PNEUMONIA'],
                yticklabels=['NORMAL', 'PNEUMONIA'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        test_gen.classes,
        y_pred,
        target_names=['NORMAL', 'PNEUMONIA']))
    
    return metrics

def plot_history(history1, history2):
    """Plot training history."""
    # Combine histories
    combined = {}
    for key in history1.history.keys():
        combined[key] = history1.history[key] + history2.history[key]
    
    epochs = range(1, len(combined['loss']) + 1)
    phase_change = len(history1.history['loss'])
    
    # Plot metrics
    metrics = ['loss', 'accuracy', 'recall', 'precision', 'auc']
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        plt.plot(epochs, combined[metric], label=f'Training {metric}')
        plt.plot(epochs, combined[f'val_{metric}'], label=f'Validation {metric}')
        plt.axvline(phase_change, color='k', linestyle='--', label='Fine-tuning start')
        plt.title(metric.upper())
        plt.xlabel('Epochs')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    """Main execution function."""
    # Prepare data
    train_gen, val_gen, test_gen, class_weights = prepare_data(
        train_dir='/Users/rosanalongares/Desktop/chest_xray/chest_xray/train',
        val_dir='/Users/rosanalongares/Desktop/chest_xray/chest_xray/val',
        test_dir='/Users/rosanalongares/Desktop/chest_xray/chest_xray/test'
    )

    # Build and train model
    model = build_model()
    model, history1, history2 = train_model(model, train_gen, val_gen, class_weights)

    # Evaluate
    test_metrics = evaluate_model(model, test_gen)
    print("\nTest Metrics:")
    for name, value in test_metrics.items():
        print(f"{name}: {value:.4f}")

    # Visualize training
    plot_history(history1, history2)

    # Save model with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model.save(f'pneumonia_effnet_{timestamp}.h5')
    print(f"\nModel saved as 'pneumonia_effnet_{timestamp}.h5'")

if __name__ == '__main__':
    main()