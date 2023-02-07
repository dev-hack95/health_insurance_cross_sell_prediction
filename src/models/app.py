import streamlit as st
import numpy as np
import pickle


model = pickle.load(open("./artifacts/model.pkl", "rb"))
scaler = pickle.load(open("./artifacts/scaler.pkl", "rb"))
######################################## Streamlit ##############################################

st.header("Health Insurance Prediction")

Gender = st.radio("Gender ", ["Male", "Female"])
Age = st.number_input("Age")
Driving_License = st.radio("Driving_License", ["Yes", "No"])
Region_Code = st.number_input("Region_Code")
Previously_Insured = st.radio("Previously_Insured ", ["Yes", "No"])
Vehicle_Age = st.selectbox("Vehicle_Age: ", ["> 2 Years", "1-2 Year", "< 1 Year"])
Vehicle_Damage = st.radio("Vehicle_Damage ", ["Yes", "No"])
Annual_Premium = st.number_input("Annual_Premium")
Policy_Sales_Channel = st.selectbox("Policy_Sales_Channel", [ 26., 152., 160., 124.,  14.,  13.,  30., 156., 163., 157., 122., 19.,  22.,  15., 154.,  16.,  52., 155.,  11., 151., 125.,  25., 61.,   1.,  86.,  31., 150.,  23.,  60.,  21., 121.,   3., 139., 12.,  29.,  55.,   7.,  47., 127., 153.,  78., 158.,  89.,  32.,  8.,  10., 120.,  65.,   4.,  42.,  83., 136.,  24.,  18.,  56., 48., 106.,  54.,  93., 116.,  91.,  45.,   9., 145., 147.,  44., 109.,  37., 140., 107., 128., 131., 114., 118., 159., 119., 105., 135.,  62., 138., 129.,  88.,  92., 111., 113.,  73.,  36.,  28., 35.,  59.,  53., 148., 133., 108.,  64.,  39.,  94., 132.,  46.,  81., 103.,  90.,  51.,  27., 146.,  63.,  96.,  40.,  66., 100., 95., 123.,  98.,  75.,  69., 130., 134.,  49.,  97.,  38.,  17., 110.,  80.,  71., 117.,  58.,  20.,  76., 104.,  87.,  84., 137., 126.,  68.,  67., 101., 115.,  57.,  82.,  79., 112.,  99.,  70.,  2.,  34.,  33.,  74., 102., 149.,  43.,   6.,  50., 144., 143., 41.])
Vintage = st.number_input("Vintage ")

if Gender == "Male" :
    Gender = 1
else:
    Gender = 0

if Driving_License == 'Yes' :
    Driving_License = 1
else:
    Driving_License = 0

if Previously_Insured == 'Yes':
    Previously_Insured = 1
else:
    Previously_Insured = 0

if Vehicle_Age == "> 2 Years" :
    Vehicle_Age = 2
elif Vehicle_Age == "1-2 Year" :
    Vehicle_Age = 0
else:
    Vehicle_Age = 1

if Vehicle_Damage == 'Yes' :
    Vehicle_Damage = 1
else:
    Vehicle_Damage = 0

data_1 = [Gender, Age, Driving_License, Region_Code, Previously_Insured, Vehicle_Age, Vehicle_Damage, Policy_Sales_Channel, Vintage]
data_2 = [Annual_Premium]
data_2 = np.reshape(data_2, (1, -1))
data_2 = scaler.transform(data_2)

for i in data_2:
    for j in i:
        data_1.insert(7, j)

data = data_1
data = np.reshape(data, (1, -1))
prediction = model.predict(data)

submit = st.button('Predict')
if submit:
    st.write('Response from company',prediction[0])