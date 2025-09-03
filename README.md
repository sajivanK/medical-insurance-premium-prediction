
# 🏥 Medical Insurance Premium Prediction

This project predicts **medical insurance premiums** based on user inputs such as age, sex, BMI, number of children, smoking status, and region.  
It uses **Machine Learning (ANN)** trained on the [Medical Insurance dataset](https://www.kaggle.com/mirichoi0218/insurance) with preprocessing and scaling for better accuracy.

---

## ✨ Features
- 📊 **Data Preprocessing**: One-hot encoding, feature scaling with `StandardScaler` (X) and `MinMaxScaler` (y).  
- 🤖 **Model**: Artificial Neural Network (ANN) built with TensorFlow/Keras.  
- 🎨 **Web App**: Interactive **Streamlit UI** with premium design.  
- 💡 **Prediction**: Instant insurance premium estimation.  

---

## 🚀 Live Demo
👉 [Try the App Here]


## 📂 Project Structure
```

├── app.py                 # Streamlit app
├── insurance\_model.keras  # Trained ANN model (or insurance\_weights.h5 if using weights)
├── scaler\_x.pkl           # Feature scaler
├── scaler\_y.pkl           # Target scaler
├── requirements.txt       # Dependencies
└── README.md              # Project documentation

````

---


## 📊 Dataset

The dataset includes the following columns:

* `age`: Age of the individual
* `sex`: Gender (male/female)
* `bmi`: Body Mass Index
* `children`: Number of children covered
* `smoker`: Smoking status (yes/no)
* `region`: Region (northeast, northwest, southeast, southwest)
* `charges`: Medical insurance premium (target variable)

---

## 📈 Model Performance

* ANN trained with:

  * Input: 8 features (age, sex, bmi, children, smoker, + one-hot encoded region)
  * Hidden Layers: 128 → 64 → 1
  * Loss: MSE
* Achieved **\~70% R² Score** on test data.

---


## 🛠️ Tech Stack

* **Python**
* **TensorFlow / Keras**
* **Scikit-learn**
* **Streamlit**
* **Pandas / NumPy**

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📜 License

This project is licensed under the MIT License.
