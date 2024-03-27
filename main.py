# This is a sample Python script.
#import pickle
import joblib as jl
import pandas as pd
import streamlit as st

st.header("Machine learning-based classification for poor social support among university students after analyzing demographics, lifestyles, exercise, and mental health status.")
st.sidebar.title("Parameters Selection Panel")
st.sidebar.markdown("Picking up parameters")

Gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
Age = st.sidebar.slider("Age", 15, 25)
Marital_status = st.sidebar.selectbox("Marital status", ("Single", "Dating", "Married"))
Low_salt = st.sidebar.selectbox("Low salt diets", ("No", "Yes"))
Vegetable = st.sidebar.selectbox("Vegetable diets", ("No", "Yes"))
Fruit = st.sidebar.selectbox("Fruit diets", ("No", "Yes"))
Sedentary_time = st.sidebar.selectbox("Sedentary time", ("< 1 hour", "≧1 and <3 hours", "≧3 and <6 hours", "≧6 hours"))
Stress_event = st.sidebar.selectbox("Life stress event", ("YES", "NO"))
Sport_frequency = st.sidebar.selectbox("Sport frequency", ("None",  "1-2", "3-4", "≧5"))
Chronic_disease = st.sidebar.selectbox("Chronic disease", ("None", "Existed"))
Electronic_time = st.sidebar.slider("Exposure time of electronic screens (h/day)", 0, 18)
SES = st.sidebar.slider("Self-esteem score", 20, 40)
PHQ9 = st.sidebar.slider("Depression score", 0, 20)
GAD7 = st.sidebar.slider("Anxiety score", 0, 20)

if st.button("Submit"):
    rf_clf = jl.load("final_roundweb.pkl")
    x = pd.DataFrame([[Gender, Marital_status, Low_salt, Vegetable, Fruit, Sedentary_time, Stress_event, Sport_frequency, Chronic_disease, Age, Electronic_time, SES, PHQ9, GAD7]],
                     columns=["Gender", "Marital_status", "Low_salt", "Vegetable", "Fruit", "Sedentary_time", "Stress_event", "Sport_frequency", "Chronic_disease", "Age", "Electronic_time", "SES", "PHQ9", "GAD7"])
    x = x.replace(["Male", "Female"], [1, 2])
    x = x.replace(["Single", "Dating", "Married"], [1, 2, 3])
    x = x.replace(["No", "Yes"], [0, 1])
    x = x.replace(["< 1 hour", "≧1 and <3 hours", "≧3 and <6 hours", "≧6 hours"], [1, 2, 3, 4])
    x = x.replace(["YES", "NO"], [1, 2])
    x = x.replace(["None",  "1-2", "3-4", "≧5"], [1, 2, 3, 4])
    x = x.replace(["None", "Existed"], [2, 1])

    # Get prediction
    prediction = rf_clf.predict_proba(x)[0, 1]
        # Output prediction
    st.text(f"Probability of Poor Social Support: {'{:.2%}'.format(round(prediction, 5))}")
    if prediction < 0.520:
        st.success(f"Risk group: low-risk group")
    else:
        st.error(f"Risk group: High-risk group")
st.subheader('Model Information')
st.markdown('The RF model shows the most promising prediction performance with the highest AUC value of 0.947 (95% CI: 0.935-0.960) in the study, and the AI application established based the RF model can be treated as a useful tool to stratify university students with high risk of suffering from poor social support.')