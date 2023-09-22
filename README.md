# Kidney Stone Prediction Using ML and DL With Deployment(Streamlit cloud)

## Problem Statement:
Kidney stones are a common health problem that affects millions of people worldwide. Early detection and timely intervention can help prevent complications and improve treatment outcomes. However, traditional methods for diagnosing kidney stones can be invasive and time-consuming. Therefore, the aim of this project is to develop a machine learning and deep learning model that can predict the presence of kidney stones in patients based on their medical history, lab reports, and other relevant factors. The model has been trained on a large dataset of patient data using advanced techniques such as Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks to achieve high accuracy. The model has been deployed on the web using Streamlit Cloud, a powerful platform for building and deploying data applications. This will allow patients and healthcare providers to easily access the model and get quick and accurate predictions about the presence of kidney stones in patients. The ultimate goal of this project is to improve the speed and accuracy of kidney stone diagnosis, leading to early precautions and diagnosis and reduced healthcare costs.

## Highlights from the project:
- Data cleansing and exploratory data analysis (EDA)
- Correlation heatmap to derive correlation amongst the features
- Using algorithms such as Logistic Regression, Decision Tree, Random Forest, Support Vector Classifier, Extreme Gradient Boosting Classifier
- Using Convolutional Neural Networks (CNN) and Bi-Long Short Time Memory (Bi-LSTM) models for deep learning based prediction
- Evaluating the models based on the performace metrics such as accuracy, precision value, recall value, f1-score
- Deploying the model on Streamlit Cloud

## Project Structure
<b> This project has four major parts :

1. model.py - This contains code fot our Machine Learning model to predict employee salaries absed on trainign data in 'hiring.csv' file.
2. app.py - This contains Flask APIs that receives employee details through GUI or API calls, computes the precited value based on our model and returns it.
3. Dataset - This contains the dataset for this project.
4. my_model.h5 - This folder contains a trained machine learning model, specifically a Keras model.

## Running the project
1. Ensure that you are in the project home directory. Create the machine learning and deep learning model by running in jupyter notebook.

   This would create a serialized Keras model into a file my_model.h5

2. Run app.py using below command to start Streamlit cloud app

   streamlit run app.py

   By default, streamlit will run on port 8501.

3. Navigate to URL http://localhost:8501
   
   We should be able to view the homepage on the HTML page! ![Image-1](https://github.com/Sidduswamy94/Kidney_Stone_Prediction_Using_ML_DL_With_Deployment/assets/119415794/c92c073c-08a3-4958-ba1c-97d0bc770825)


   Enter valid numerical values in all 6 input boxes and hit Predict.

   If everything goes well, we should be able to see the kidney stone prediction on the HTML page! 
