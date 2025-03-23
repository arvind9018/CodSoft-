# Fraud Detection using Machine Learning & Deep Learning

This project identifies fraudulent credit card transactions using **Random Forest (ML)** and **LSTM (Deep Learning)** models. It includes data preprocessing, model training, and an API for predictions.

---

## 📌 Project Features
✅ **Data Preprocessing** (Scaling, SMOTE for imbalance handling)  
✅ **Machine Learning Model** (Random Forest)  
✅ **Deep Learning Model** (LSTM)  
✅ **Flask API Deployment** for real-time predictions  
✅ **Testing via Postman & cURL**  

---

## 📂 Project Structure
```
├── preprocess.py        # Data Preprocessing Script
├── train_ml.py          # Random Forest Training
├── train_lstm.py        # LSTM Training
├── app.py               # Flask API
├── requirements.txt     # Python Dependencies
├── README.md            # Project Documentation
├── X_train.npy, X_test.npy, y_train.npy, y_test.npy  # Preprocessed Data
├── rf_model.pkl         # Saved Random Forest Model
└── lstm_model.h5        # Saved LSTM Model
```

---

## 🔧 Installation
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-repo/fraud-detection.git
cd fraud-detection
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Download Dataset**
- Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Place it in the project folder

---

## 🚀 Running the Project

### **Step 1: Data Preprocessing**
```bash
python preprocess.py
```
✅ **Generates:** `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`

### **Step 2: Train Models**
#### **Train Random Forest**
```bash
python train_ml.py
```
✅ **Saves:** `rf_model.pkl`

#### **Train LSTM Model**
```bash
python train_lstm.py
```
✅ **Saves:** `lstm_model.h5`

### **Step 3: Run Flask API**
```bash
python app.py
```
✅ **API Running at:** `http://127.0.0.1:5000/predict`

---

## 📌 Testing the API

### **1️⃣ Using Postman**
1. Open **Postman**
2. **POST Request** to: `http://127.0.0.1:5000/predict`
3. Go to **Body** → **raw** → **JSON**, and enter:
   ```json
   {
     "features": [0.2, -1.1, 0.5, 1.8, -0.9, 2.7, -1.8, 0.6, -0.5, 0.9, 0.3, -0.7, 1.2, 0.4, -1.3]
   }
   ```
4. Click **Send** and receive a response like:
   ```json
   {
     "RandomForest_Prediction": "Genuine",
     "LSTM_Prediction": "Fraud"
   }
   ```

### **2️⃣ Using cURL (Command Line)**
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"features": [0.2, -1.1, 0.5, 1.8, -0.9, 2.7, -1.8, 0.6, -0.5, 0.9, 0.3, -0.7, 1.2, 0.4, -1.3]}'
```
✅ Expected Output:
```json
{
  "RandomForest_Prediction": "Genuine",
  "LSTM_Prediction": "Fraud"
}
```

---

## 📡 Deployment
To deploy the API, use **Render, AWS, or Google Cloud**. Example for **Render**:
1. Push code to GitHub
2. Connect GitHub repo to **Render.com**
3. Deploy as a **Flask Web Service**

---

## 🎯 Future Improvements
🔹 **Optimize Model (XGBoost, Autoencoders)**  
🔹 **Deploy Frontend (React.js + Flask)**  
🔹 **Integrate into a Real-Time Monitoring System**  

---

## 🤝 Contributing
Feel free to submit issues or pull requests to improve this project!

**🚀 Happy Coding!** 🎉
