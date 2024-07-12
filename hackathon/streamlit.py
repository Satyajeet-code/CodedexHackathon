import numpy as np 
import tensorflow as tf
import streamlit as st
from dictionary_data import discipline_dict

st.set_page_config(page_title="Olympic Rank Prediction")
st.header("Olympic Rank Prediction 🏸🏐🤖")


def load_model():
    model = tf.keras.models.load_model("ann.h5")
    return model


def predict_rank(model, features):
    data_array = np.array(features)
    y_pred_probs = model.predict(data_array)

    y_pred = y_pred_probs.argmax(axis=1)
    return y_pred[0]


model = load_model()

options = discipline_dict.keys()
options=list(options)
st.write("Use sliders to enter the value. Use arrow keys for max precision")
gold = st.slider("Total Gold medals:", 0, 60, 30)
silver = st.slider("Total Silver medals:", 0, 60, 30)
bronze = st.slider("Total Bronze medals:", 0, 60, 30)
total = gold+silver+bronze
female = st.slider("Total females participated:", 0, 970, 30)
male = st.slider("Total males participated:", 0, 1080, 30)
selected_option = st.selectbox("Select the discipline:", options)
discipline_encoded=options = discipline_dict[selected_option]

st.write("The total number of medals: ", total)


submit = st.button("Predict Rank")

if submit:
    features = [gold, silver, bronze,total, female, male, discipline_encoded]
    rank_prediction = predict_rank(model, [features])
    st.subheader("Predicted Rank:")
    st.write("With the current data provided your predicted rank is: ", rank_prediction)
    if rank_prediction>10 and rank_prediction<=50:
        st.write("Get more medals and try to get to top 10 💪" )
    elif rank_prediction>50:
        st.write("Get more medals to improve your ranks 😁" )
    else:
        st.write("Congrats you are in top 10 🥳" )


    st.markdown("Made with ❤️ by *Sabudana 🍈*")