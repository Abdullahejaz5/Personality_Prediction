# 🧠 Personality Prediction using Machine Learning (Logistic Regression)

This project is a full **Machine Learning web app** built with **Streamlit** that predicts whether a person is an **Introvert** or **Extrovert** based on their behavior and social traits.  
It covers the complete **ML pipeline** — from **data analysis** to **model training** and **live predictions**.

---

## 📁 Files Included

| File                      | Description                                         |
|--------------------------|-----------------------------------------------------|
| `app.py`                 | Streamlit app with EDA, ML model, and predictions  |
| `personality_dataset.csv`| Dataset containing social and behavioral features  |

---

## 🚀 Features

- ✅ Clean and interactive **Streamlit UI**
- 📊 Full **Exploratory Data Analysis (EDA)**
- 🔥 Correlation Heatmaps, Histograms, Boxplots, and Pairplots
- 🤖 **Machine Learning Model**: Logistic Regression
- 📈 Shows **Accuracy**, **F1 Score**, and **Classification Report**
- 🧪 Make **real-time predictions** with your own input

---

## 🧠 ML Model Overview

This app uses a **Logistic Regression** model to predict personality based on these features:

- `Time_spent_Alone`
- `Stage_fear` (Yes/No)
- `Social_event_attendance`
- `Going_outside`
- `Drained_after_socializing` (Yes/No)
- `Friends_circle_size`
- `Post_frequency` on social media

🔧 Data is cleaned, missing values are handled, and numerical columns are normalized using **Min-Max Scaling** before training the model.

---

## 📊 Dataset Description

The dataset contains behavioral data of individuals and a target column `Personality` with values:  
- **Introvert**  
- **Extrovert**

---

## 📦 Libraries Used

- `pandas`, `numpy` – for data handling  
- `seaborn`, `matplotlib` – for visualizations  
- `scikit-learn` – for model building  
- `scipy` – for z-score/outlier detection  
- `streamlit` – to build the interactive web interface

---

## ▶️ How to Run the App

1. **Install dependencies** (if not already installed):
   ```bash
   pip install streamlit pandas numpy scikit-learn matplotlib seaborn scipy
