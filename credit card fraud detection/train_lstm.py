import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load data
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

# Reshape for LSTM (samples, timesteps, features)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Define LSTM model
lstm_model = Sequential([
    LSTM(64, input_shape=(1, X_train.shape[2]), activation='relu', return_sequences=True),
    LSTM(32, activation='relu', return_sequences=False),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
lstm_model.fit(X_train, y_train, epochs=10, batch_size=256, validation_split=0.1)

# Save model
lstm_model.save("lstm_model.h5")

# Evaluate model
y_pred_lstm = (lstm_model.predict(X_test) > 0.5).astype(int)
from sklearn.metrics import classification_report
print("LSTM Model Performance:\n", classification_report(y_test, y_pred_lstm))
