# frontend/streamlit_app.py

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import KBinsDiscretizer

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Load the original dataset to get the exact categories
df = pd.read_csv("Student_survey.csv")
df = df.rename(columns={"Affects_Academic_Performance": "Affects"})
df = df[df["Affects"].notna()]

# Get unique values for each categorical column
countries = sorted(df['Country'].unique())
platforms = sorted(df['Most_Used_Platform'].unique())
academic_levels = sorted(df['Academic_Level'].unique())
genders = sorted(df['Gender'].unique())

# Create a sample row with all possible categories
sample_data = []
for gender in genders:
    for level in academic_levels:
        for country in countries:
            for platform in platforms:
                sample_data.append({
                    "Age": 20,
                    "Gender": gender,
                    "Academic_Level": level,
                    "Country": country,
                    "Avg_Daily_Usage_Hours": 4.5,
                    "Most_Used_Platform": platform,
                    "Sleep_Hours_Per_Night": 6.5,
                    "Mental_Health_Score": 6,
                    "Sleep_Usage_Interaction": 29.25,
                    "Mental_Usage_Interaction": 27,
                    "Age_Binned": 0,
                    "Usage_Binned": 0,
                    "Sleep_Binned": 0
                })

# Create sample DataFrame and get feature order
sample_df = pd.DataFrame(sample_data)
X_sample = pd.get_dummies(sample_df, columns=["Gender", "Academic_Level", "Country", "Most_Used_Platform"])
feature_order = X_sample.columns.tolist()

st.set_page_config(page_title="Academic Impact Predictor")

st.title("📘 Social Media Academic Impact Predictor")
st.markdown("This app predicts whether your social media use negatively affects your academic performance based on your habits.")

# Input fields
age = st.slider("Age", 18, 30, 20)
gender = st.selectbox("Gender", genders)
academic_level = st.selectbox("Academic Level", academic_levels)
country = st.selectbox("Country", countries)
avg_usage = st.slider("Average Daily Social Media Usage (Hours)", 1.5, 8.5, 4.5, step=0.1)
most_used = st.selectbox("Most Used Platform", platforms)
sleep_hours = st.slider("Sleep Hours per Night", 3.5, 9.5, 6.5, step=0.1)
mental_score = st.slider("Mental Health Score (1 = Poor, 10 = Excellent)", 1, 10, 6)

# Calculate interaction features
sleep_usage_interaction = sleep_hours * avg_usage
mental_usage_interaction = mental_score * avg_usage

# Bin continuous variables
discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
age_binned = discretizer.fit_transform([[age]])[0][0]
usage_binned = discretizer.fit_transform([[avg_usage]])[0][0]
sleep_binned = discretizer.fit_transform([[sleep_hours]])[0][0]

# Create input data
input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Academic_Level": [academic_level],
    "Country": [country],
    "Avg_Daily_Usage_Hours": [avg_usage],
    "Most_Used_Platform": [most_used],
    "Sleep_Hours_Per_Night": [sleep_hours],
    "Mental_Health_Score": [mental_score],
    "Sleep_Usage_Interaction": [sleep_usage_interaction],
    "Mental_Usage_Interaction": [mental_usage_interaction],
    "Age_Binned": [age_binned],
    "Usage_Binned": [usage_binned],
    "Sleep_Binned": [sleep_binned]
})

# One-hot encode the input data
X = pd.get_dummies(input_data, columns=["Gender", "Academic_Level", "Country", "Most_Used_Platform"])

# Ensure all features from training are present
for feature in feature_order:
    if feature not in X.columns:
        X[feature] = 0

# Reorder columns to match training data
X = X[feature_order]

# Scale numeric columns
numeric_cols = ["Age", "Avg_Daily_Usage_Hours", "Sleep_Hours_Per_Night", "Mental_Health_Score",
                "Sleep_Usage_Interaction", "Mental_Usage_Interaction"]
X[numeric_cols] = scaler.transform(X[numeric_cols])

# Predict button
if st.button("Predict"):
    # Make prediction
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    
    # Calculate risk factors
    risk_factors = []
    if avg_usage > 5:
        risk_factors.append(("High social media usage", "Consider setting daily limits"))
    if sleep_hours < 6:
        risk_factors.append(("Insufficient sleep", "Aim for 7-8 hours of sleep"))
    if mental_score < 5:
        risk_factors.append(("Low mental health score", "Consider seeking support"))
    if avg_usage > 3 and sleep_hours < 7:
        risk_factors.append(("Sleep-usage imbalance", "Reduce evening screen time"))
    
    # Calculate positive factors
    positive_factors = []
    if avg_usage <= 3:
        positive_factors.append(("Moderate social media usage", "Good balance"))
    if sleep_hours >= 7:
        positive_factors.append(("Good sleep habits", "Maintain this routine"))
    if mental_score >= 7:
        positive_factors.append(("Good mental health", "Keep up self-care practices"))
    
    # Display prediction result with context
    st.write("### Prediction Analysis")
    st.write(f"Confidence: {probabilities[1]*100:.1f}%")
    
    if prediction == 1:
        st.error("⚠️ The model predicts your social media use is **likely** affecting your academic performance.")
        
        st.write("#### Identified Risk Factors:")
        for factor, suggestion in risk_factors:
            st.write(f"- {factor}: {suggestion}")
            
        st.write("#### Recommendations:")
        st.write("1. Set specific time limits for social media use")
        st.write("2. Create a study schedule with regular breaks")
        st.write("3. Use apps to track and limit screen time")
        st.write("4. Establish a consistent sleep routine")
        
    else:
        st.success("✅ The model predicts your social media use is **not likely** affecting your academic performance.")
        
        st.write("#### Positive Factors in Your Profile:")
        for factor, note in positive_factors:
            st.write(f"- {factor}: {note}")
            
        st.write("#### Suggestions for Maintaining Balance:")
        st.write("1. Continue monitoring your social media usage")
        st.write("2. Maintain your current sleep schedule")
        st.write("3. Regular self-assessment of academic performance")
        st.write("4. Stay aware of any changes in study habits")
    
    # Add disclaimer
    st.write("---")
    st.write("""
    **Note**: This prediction is based on statistical patterns and should be used as a guide only. 
    Individual experiences may vary, and other factors not captured in this model may also affect academic performance.
    """)
