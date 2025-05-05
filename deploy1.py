import streamlit as st
import numpy as np
import joblib

# Load the trained SVM model and label encoder
model = joblib.load('svm_model.pkl')
le = joblib.load('label_encoder.pkl')

# Function to take numeric input and make prediction
def predict_iris_species(sepal_length, sepal_width, petal_length, petal_width):
    # Create feature array
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Predict using the model
    predicted_class = model.predict(features)

    # Decode label
    predicted_species = le.inverse_transform(predicted_class)

    return predicted_species[0]

# -------- Streamlit UI --------
st.title("Iris Flower Species Prediction")

# User input section
st.sidebar.header("Input Parameters")
sepal_length = st.sidebar.number_input("Sepal Length", min_value=0.0, value=5.1, step=0.1)
sepal_width = st.sidebar.number_input("Sepal Width", min_value=0.0, value=3.5, step=0.1)
petal_length = st.sidebar.number_input("Petal Length", min_value=0.0, value=1.4, step=0.1)
petal_width = st.sidebar.number_input("Petal Width", min_value=0.0, value=0.2, step=0.1)

# Button to trigger prediction
if st.sidebar.button("Predict"):
    result = predict_iris_species(sepal_length, sepal_width, petal_length, petal_width)
    st.subheader(f"Predicted Iris Species: {result}")

# Optionally, you can add additional information about the model or dataset
st.markdown("""
This model predicts the species of Iris flowers based on their physical attributes: 
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The trained model uses a Support Vector Machine (SVM) to classify the species into one of the following categories:
- Iris-setosa
- Iris-versicolor
- Iris-virginica
""")
