from flask import Flask, request, jsonify
import numpy as np
import joblib
import tensorflow as tf

app = Flask(__name__)

# Load models
rf_model = joblib.load("rf_model.pkl")
lstm_model = tf.keras.models.load_model("lstm_model.h5")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    data = np.array(data).reshape(1, -1)  # Reshape for Random Forest

    # Predict with both models
    rf_pred = rf_model.predict(data)[0]
    data_lstm = data.reshape(1, 1, data.shape[1])  # Reshape for LSTM
    lstm_pred = (lstm_model.predict(data_lstm) > 0.5).astype(int)[0][0]

    return jsonify({
        "RandomForest_Prediction": "Fraud" if rf_pred == 1 else "Genuine",
        "LSTM_Prediction": "Fraud" if lstm_pred == 1 else "Genuine"
    })

if __name__ == '__main__':
    app.run(debug=True)
