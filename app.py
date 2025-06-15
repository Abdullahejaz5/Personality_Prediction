import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy.stats import zscore

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("personality_dataset.csv")
    return df

df = load_data()

# App title
st.title("Personality Prediction Project")
st.markdown("A data-driven approach to predicting personality types based on behavioral and social attributes.")

# Sidebar navigation
section = st.sidebar.radio("Navigation", ["Introduction", "EDA", "Model", "Conclusion"])

# ------------------------- INTRODUCTION SECTION -----------------------------
if section == "Introduction":
    st.header("📌 Introduction")
    st.markdown("""
    This project aims to explore and predict **personality types** (Introvert vs Extrovert) using a variety of features like:
    - Time spent alone
    - Stage fear
    - Social event attendance
    - Frequency of going outside
    - Feeling drained after socializing
    - Friends circle size
    - Post frequency on social media
    
    The final model is a **Logistic Regression classifier**, and this app will walk you through the dataset exploration, preprocessing, model performance, and results.
    """)

# ------------------------- EDA SECTION -----------------------------
elif section == "EDA":
    st.header("🔍 Exploratory Data Analysis")

    
    
    st.subheader("Dataset Snapshot")
    st.dataframe(df.head())

    st.subheader("Descriptive Statistics")
    st.write(df.describe())

    st.subheader("Number of unique values")
    st.dataframe(df.nunique())

    st.subheader("DataTypes")
    st.dataframe(df.dtypes)

    st.subheader("Grouped Averages by Personality:")
    st.dataframe(df.groupby('Personality').mean(numeric_only=True))

    st.subheader("Missing Values")
    st.write(df.isna().sum())

    z_scores = np.abs(zscore(df.select_dtypes(include=np.number)))
    outliers = (z_scores > 3).any(axis=1)

    st.subheader("Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    st.subheader(f"\nNumber of potential outliers: {outliers.sum()}")

    st.subheader("Histograms of Features")
    selected_col = st.selectbox("Select a column to view distribution", df.columns[:-1])
    fig, ax = plt.subplots()
    sns.histplot(df[selected_col], kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Boxplot for Numeric Features")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df.select_dtypes(include=np.number), ax=ax)
    st.pyplot(fig)

    st.subheader("Pairwise Relationships")
    st.markdown("Due to performance, only shows for a sample of the dataset.")
    sample_df = df.sample(min(200, len(df)))
    fig = sns.pairplot(sample_df, hue="Personality")
    st.pyplot(fig)
    
    

# ------------------------- MODEL SECTION -----------------------------
elif section == "Model":
    st.header("🤖 Model Training & Prediction")
    maxmin=dict()
    # Preprocessing
    def minmax(col):
        return (col - col.min()) / (col.max() - col.min()), col.min(), col.max()

    
    for col in ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']:
        df[col] = df[col].fillna(df[col].mean())
    df['Stage_fear'] = df['Stage_fear'].fillna(df['Stage_fear'].mode().iloc[0])
    df['Drained_after_socializing'] = df['Drained_after_socializing'].fillna(df['Drained_after_socializing'].mode().iloc[0])
    for col in ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']:
        df[col], m1, m2 = minmax(df[col])
        maxmin[col]=(m2,m1)
    df = pd.get_dummies(df, columns=['Stage_fear', 'Drained_after_socializing', 'Personality'], drop_first=True, dtype=int)

    y = df['Personality_Introvert']
    X = df.drop('Personality_Introvert', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("Model Summary")
    st.write("**Model Used:** Logistic Regression")
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
    st.write(f"**F1 Score:** {f1_score(y_test, y_pred):.2f}")
    
    st.text("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:.0f}"}))

    st.subheader("_Make Your Own Prediction_")

    input_data = {}
    for col in X.columns:
        if X[col].nunique() <= 2:
            take= st.selectbox(f"{col}", ['YES', 'NO'])
            if take=='YES':
                input_data[col]=1
            else:
                input_data[col]=0
        else:
            take1=maxmin[col]
            take2= st.slider(f"{col}", int(take1[1]), int(take1[0]), int(2))
            input_data[col] =(take2-take1[1])/(take1[0]-take1[1])
    if st.button("Predict Personality"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        result = "Introvert" if prediction == 1 else "Extrovert"
        st.success(f"Predicted Personality: **{result}**")

# ------------------------- CONCLUSION SECTION -----------------------------
elif section == "Conclusion":
    st.header("📌 Conclusion")
    st.markdown("""
    ### Key Takeaways:
    - Logistic Regression can reasonably predict whether someone is an introvert or extrovert.
    - Features like time spent alone and social event attendance show good correlation with personality.
    - This kind of analysis can help in mental health assessment, team building, or personalized experiences.
    
    #### Future Improvements:
    - Use more advanced models like Random Forest or XGBoost
    - Expand dataset and include more demographic features
    - Deploy the model via an API for integration with real apps
    """)
