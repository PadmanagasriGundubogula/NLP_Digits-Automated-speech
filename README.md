# Digits-Automated-speech-Recogntion-b4
# Speech Digit Recognition using CNN and MFCC Features with SpaCy

## Overview

This project performs speech digit recognition using a **Convolutional Neural Network (CNN)** with **MFCC features** extracted from audio files. An optional **SpaCy** step is included for text processing, which can be used to analyze speech-to-text data (if applicable).

The primary components of this project are:
1. **CNN Model**: Trained on MFCC features to recognize digits from speech.
2. **Data Preprocessing**: Extracts MFCC features from the audio data.
3. **Inference**: Runs predictions on unseen audio files and visualizes results.
4. **Workflow Visualization**: Generated using `graphviz` to represent the model workflow.

## Folder Structure


## Prerequisites

Before running the code, make sure to install the following dependencies:

- `numpy`
- `librosa`
- `matplotlib`
- `tensorflow`
- `spacy`
- `scikit-learn`
- `graphviz`
  
You can install these using `pip`:
pip install numpy librosa matplotlib tensorflow spacy scikit-learn graphviz

Additionally, you need to download the SpaCy model (if using text processing):
python -m spacy download en_core_web_sm
Instructions to Run the Code
Step 1: Prepare the Dataset
Place your audio files (in .wav format) in the data/recordings/ folder. The filenames should follow the format <digit>_<random_string>.wav, where <digit> represents the digit to be recognized (0-9).

Step 2: Data Preprocessing
Run the data_preprocessing.py script to load and extract MFCC features from the audio data.


python Codes/data_preprocessing.py

Step 3: Train the Model
To train the CNN model using the preprocessed MFCC features, run the model.py script:
python Codes/model.py


Step 4: Inference
To run inference on new audio files, use the inference.py script. It will load the trained model and make predictions on the provided audio files.
python Codes/inference.py

Step 5: Workflow Visualization
To generate the workflow diagram that visualizes the steps in the project, run the following:
python Codes/workflow_graph.py


Google Drive Links for Model Files
All model files and intermediate files are stored on Google Drive. Please find the links below:
https://drive.google.com/drive/folders/1Diy687WQ4aomtxzWdoOo7lLppZIIHnip?usp=sharing

program code:
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import spacy

# Load the SpaCy model (assuming you're working with English text)
nlp = spacy.load("en_core_web_sm")

# Step 1: Load Dataset
def load_dataset(data_path):
    """
    Load audio files and extract MFCC features and corresponding labels.
    """
    X, y = [], []
    files = os.listdir(data_path)
    for file in files:
        if file.endswith('.wav'):
            file_path = os.path.join(data_path, file)
            audio, sr = librosa.load(file_path, sr=None)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfcc = np.mean(mfcc.T, axis=0)
            X.append(mfcc)
            label = int(file.split('_')[0])  # Extract digit from filename
            y.append(label)
    return np.array(X), np.array(y), files

data_path = "/content/drive/MyDrive/free-spoken-digit-dataset-master/recordings"
X, y, file_names = load_dataset(data_path)

# Normalize Features
X = (X - np.mean(X)) / np.std(X)
y = to_categorical(y, num_classes=10)

# Split Data
X_train, X_test, y_train, y_test, train_files, test_files = train_test_split(X, y, file_names, test_size=0.2, random_state=42)

# Reshape for CNN Input
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Step 2: Build and Train Model
model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(13, 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Use EarlyStopping and ModelCheckpoint callbacks to prevent overfitting and save the best model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint("best_model.keras", monitor='val_loss', save_best_only=True)
 # Corrected line

# Training with callbacks
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])

# Save Model
model.save("speech_digit_recognition_model.h5")

# Evaluate the Model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

# Step 3: Visualize Predictions and Waveforms
def visualize_predictions(data_path, model, files, test_files):
    """
    Visualize audio waveforms, the predicted digits, and confidence scores.
    """
    print("\n--- Visualizing Predictions ---\n")
    for file_name in test_files[:5]:  # Visualize the first 5 test samples
        file_path = os.path.join(data_path, file_name)

        # Load audio and extract features
        audio, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc = np.mean(mfcc.T, axis=0).reshape(1, 13, 1)

        # Predict digit
        prediction = model.predict(mfcc)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # Display prediction details
        print(f"File: {file_name} | Predicted Digit: {predicted_digit} | Confidence: {confidence:.2f}%")

        # Plot waveform
        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(audio, sr=sr)
        plt.title(f"Predicted Digit: {predicted_digit} (Confidence: {confidence:.2f}%)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.show()

# Load the best model (saved during training)
model = load_model("best_model.keras")


# Visualize Predictions
visualize_predictions(data_path, model, file_names, test_files)

# Step 4: Display Evaluation Metrics (Optional)
y_pred = model.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)

# Classification report
print("\n--- Classification Report ---")
print(classification_report(np.argmax(y_test, axis=1), y_pred_class))

# Confusion matrix
print("\n--- Confusion Matrix ---")
print(confusion_matrix(np.argmax(y_test, axis=1), y_pred_class)) 

Evaluation Metrics
After training the model, the following metrics will be printed to evaluate its performance:


--- Classification Report ---
              precision    recall  f1-score   support

           0       0.92      0.84      0.88        64
           1       0.95      0.98      0.97        57
           2       0.93      0.88      0.90        57
           3       0.79      0.90      0.84        42

    accuracy                           0.90       220
   macro avg       0.90      0.90      0.90       220
weighted avg       0.90      0.90      0.90       220


--- Confusion Matrix ---
[[54  3  1  6]
 [ 1 56  0  0]
 [ 3  0 50  4]
 [ 1  0  3 38]]


 Notes
The project uses MFCC for feature extraction from audio.
The model is a 1D Convolutional Neural Network designed for digit recognition from spoken digits.
SpaCy is used for optional text analysis if needed (e.g., processing speech-to-text results).
The model's best weights are saved in the trained_models/ folder as best_model.keras.
