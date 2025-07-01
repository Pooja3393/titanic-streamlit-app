import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("titanic_model.pkl")

st.title("Titanic Survival Prediction")

# User input section
st.header("Passenger Details")
Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Sex = st.selectbox("Sex", ["male", "female"])
Age = st.slider("Age", 0, 100, 25)
SibSp = st.number_input("Number of Siblings/Spouses Aboard", 0, 10, 0)
Parch = st.number_input("Number of Parents/Children Aboard", 0, 10, 0)
Fare = st.number_input("Fare", 0.0, 600.0, 50.0)
Embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

# Convert categorical inputs
sex_map = {"male": 0, "female": 1}
embarked_map = {"S": 0, "C": 1, "Q": 2}

input_data = pd.DataFrame([{
    "Pclass": Pclass,
    "Sex": sex_map[Sex],
    "Age": Age,
    "SibSp": SibSp,
    "Parch": Parch,
    "Fare": Fare,
    "Embarked": embarked_map[Embarked]
}])

# Predict button
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("🎉 The passenger would have SURVIVED.")
    else:
        st.error("☠️ The passenger would NOT have survived.")
