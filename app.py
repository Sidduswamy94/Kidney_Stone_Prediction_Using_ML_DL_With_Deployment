import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample default values
gravity = 0.0
ph = 0.0
osmo = 0.0
cond = 0.0
urea = 0.0
calc = 0.0

# Load your Bi-LSTM model
model = load_model('my_model.h5')

# Sample DataFrame (replace this with your actual dataset)
df1 = pd.DataFrame({
    'gravity': [gravity],
    'ph': [ph],
    'osmo': [osmo],
    'cond': [cond],
    'urea': [urea],
    'calc': [calc]
})

# Function to preprocess user input
def preprocess_input(gravity, ph, osmo, cond, urea, calc):
    # Your preprocessing code here
    # Calculate osmo_cond_ratio
    osmo_cond_ratio = osmo / cond

    # Calculate urea_calc_diff
    urea_calc_diff = urea - calc

    # Standardize the values
    mean = df1[['gravity', 'ph', 'osmo', 'cond', 'urea', 'calc']].mean()
    std = df1[['gravity', 'ph', 'osmo', 'cond', 'urea', 'calc']].std()

    gravity = (gravity - mean['gravity']) / std['gravity']
    ph = (ph - mean['ph']) / std['ph']
    osmo = (osmo - mean['osmo']) / std['osmo']
    cond = (cond - mean['cond']) / std['cond']
    urea = (urea - mean['urea']) / std['urea']
    calc = (calc - mean['calc']) / std['calc']

    # Calculate osmo_urea_interaction
    osmo_urea_interaction = osmo * urea

    # Categorize the values into bins with 'drop' duplicates
    gravity_bin = pd.cut([gravity], bins=5, labels=False, duplicates='drop')
    ph_bin = pd.cut([ph], bins=5, labels=False, duplicates='drop')
    osmo_bin = pd.cut([osmo], bins=5, labels=False, duplicates='drop')
    cond_bin = pd.cut([cond], bins=5, labels=False, duplicates='drop')
    urea_bin = pd.cut([urea], bins=5, labels=False, duplicates='drop')
    calc_bin = pd.cut([calc], bins=5, labels=False, duplicates='drop')

    # Return the preprocessed input as a list
    return [gravity, ph, osmo, cond, urea, calc, osmo_cond_ratio, urea_calc_diff, osmo_urea_interaction, gravity_bin[0],
            ph_bin[0], osmo_bin[0], cond_bin[0], urea_bin[0], calc_bin[0]]



# Streamlit app
st.title("Kidney Stone Prediction")

# Input form
st.sidebar.header("Enter Values:")
gravity = st.sidebar.number_input("Gravity", min_value=0.0, max_value=100.0, value=0.0)
ph = st.sidebar.number_input("pH", min_value=0.0, max_value=14.0, value=0.0)
osmo = st.sidebar.number_input("Osmo", min_value=0.0, max_value=1000.0, value=0.0)
cond = st.sidebar.number_input("Cond", min_value=0.0, max_value=1000.0, value=0.0)
urea = st.sidebar.number_input("Urea", min_value=0.0, max_value=1000.0, value=0.0)
calc = st.sidebar.number_input("Calc", min_value=0.0, max_value=1000.0, value=0.0)

# Prediction
if st.sidebar.button("Predict"):
    # Preprocess the input
    user_input = preprocess_input(gravity, ph, osmo, cond, urea, calc)

    # Reshape and pad input for the model
    user_input = np.array(user_input).reshape(1, -1)
    user_input = pad_sequences(user_input, maxlen=15, dtype='float32')

    # Make prediction
    prediction = model.predict(user_input)

    # Display prediction result
    if prediction[0][0] > 0.1:
        st.sidebar.success("Prediction: You may have a kidney stone.")
    else:
        st.sidebar.success("Prediction: You may not have a kidney stone.")




