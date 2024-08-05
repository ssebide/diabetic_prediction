# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 23:28:54 2024

@author: Johnson
"""

import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model = pickle.load(open('C:/Johnson/Python projects/models/trained_model.sav', 'rb'))

#creating a function for prediction

def diabetes_prediction(input_data):
    
    #change the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the array as we predict for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else: 
      return 'The person is diabetic'
  
    

def main():
    
    #giving a title
    st.title('Diabetes Prediction')
    
    #getting the input data from the user
    Pregnancies = st.text_input('Number of pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin level')
    BMI = st.text_input('Body Mass Index value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree function')
    Age = st.text_input('Age')
    
    #code for prediction
    diagnosis = ''
    
    #creating a button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()
    