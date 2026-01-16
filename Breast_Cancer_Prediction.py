# lIbraries used in project ("Breast Cancer Prediction model")

import streamlit as st
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")



# Logistic Regression Model Traning
def train_model():
    data = sklearn.datasets.load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=10000)
    model.fit(X_train_scaled, y_train)

    train_accuracy = accuracy_score(y_train, model.predict(X_train_scaled))
    test_accuracy = accuracy_score(y_test, model.predict(X_test_scaled))

    return model, scaler, data.feature_names, train_accuracy, test_accuracy



# Model loading
model, scaler, feature_names, train_acc, test_acc = train_model()



# Web app implementation (Streamlit)
st.title("🩺 Breast Cancer Prediction from Features")
st.markdown("""
Provide input features to predict if a tumor is 'Benign' or 'Malignant'.
""")



# Input button
input_string = st.text_area("Provide input fearures:", height=100)


# Predict button
if st.button("Predict"):
    try:
        input_values = [float(val.strip()) for val in input_string.split(",")]

        if len(input_values) != 30:
            st.error(f" You entered {len(input_values)} values. Please provide all 30 features.")
        else:
            input_array = np.array(input_values).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)[0]

            st.success("Prediction Result:")
            if prediction == 0:
                st.error(" The tumor is 'Malignant' (Cancerous)")
                st.image("https://cdn-icons-png.flaticon.com/512/564/564619.png", width=200, caption="Malignant Tumor Detected")
            else:
                st.info("The tumor is 'Benign' (Non-cancerous)")
                st.image("https://cdn-icons-png.flaticon.com/512/190/190411.png", width=200, caption="Benign Tumor - No worries!")

    except ValueError:
        st.error(" Invalid input! Make sure you entered only numeric values.")



# Displaying Accuracy
st.markdown("---")
st.markdown(f" 'Model Accuracy:'  \n Training Accuracy: `{train_acc*100:.2f}%`  \n Testing Accuracy: `{test_acc*100:.2f}%`")
