import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import pickle


st.title("Titanic Survival Prediction")

pclass=st.slider("Passenger Class (1, 2, or 3)", min_value=1, max_value=3, value=1, step=1, key="pclass")

sex=st.selectbox("Sex", options=["male", "female"], key="sex")

sibsp=st.slider("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0, step=1, key="sibsp")

parch=st.slider("Parents/Children Aboard", min_value=0, max_value=6, value=0, step=1, key="parch")

fare=st.number_input("Fare", min_value=0.0, value=32.2, step=0.1, key="fare")

embarked=st.selectbox("Port of Embarkation", options=["Cherbourg", "Queenstown", "Southampton"], key="embarked")

data=pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Embarked": [embarked]
})

model=load_model("titanic_model.h5")

with (open("scalar.pkl", "rb") as f):
    scaler=pickle.load(f)

with (open("onehot_encoder.pkl", "rb") as f):
    onehot=pickle.load(f)

with (open("label_encoder.pkl", "rb") as f):
    label=pickle.load(f)  

data["Sex"]=label.transform(data[["Sex"]])
embarked=onehot.transform(data[["Embarked"]])
embarked=pd.DataFrame(embarked,columns=onehot.get_feature_names_out())
data = pd.concat([data.drop(columns=["Embarked"]), embarked], axis=1)
data[["Pclass","SibSp","Parch","Fare"]]=scaler.transform(data[["Pclass","SibSp","Parch","Fare"]])

y=model.predict(data)
y=y[0][0]

def Chance(y):
    if y>0.5:
        return "The passenger is likely to survive."
    else:
        return "The passenger is likely to not survive."
if st.button("Predict survival chance"):
    st.write(y)
    st.write(Chance(y))
