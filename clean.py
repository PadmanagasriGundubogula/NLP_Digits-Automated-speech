import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

DATASET_PATH = "./recordings"


def load_audio_files(dataset_path):
    audio_data = []
    labels = []
    for file_name in os.listdir(dataset_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(dataset_path, file_name)
            audio, sr = librosa.load(file_path, sr=None)
            label = int(file_name.split('_')[0])
            audio_data.append(audio)
            labels.append(label)
    return audio_data, labels


audio_data, labels = load_audio_files(DATASET_PATH)
print(f"Loaded {len(audio_data)} audio samples.")


def extract_features(audio_data, sr=22050, n_mfcc=13):
    features = []
    for audio in audio_data:
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc.T, axis=0)  # Average across time
        features.append(mfcc_mean)
    return np.array(features)

features = extract_features(audio_data)
print(f"Extracted features shape: {features.shape}")


scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


encoder = OneHotEncoder(sparse=False)
labels_onehot = encoder.fit_transform(np.array(labels).reshape(-1, 1))


X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels_onehot, test_size=0.2, random_state=42)


class_weights = {i: 1.0 for i in range(10)}

# Build a more complex neural network
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax') 
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, class_weight=class_weights, verbose=1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

# Confusion matrix
cm = confusion_matrix(y_true_labels, y_pred_labels)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification report
print(classification_report(y_true_labels, y_pred_labels))


def predict_digit(audio_path):
    audio, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)
    mfcc_mean_scaled = scaler.transform(mfcc_mean)
    prediction = model.predict(mfcc_mean_scaled)
    predicted_digit = np.argmax(prediction)
    return predicted_digit


test_audio_path = "./record_out.wav" 
predicted_digit = predict_digit(test_audio_path)
print(f"Predicted Digit: {predicted_digit}")
