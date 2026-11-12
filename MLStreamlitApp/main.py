# goal: create Streamlit app that takes a either an example dataset or an uploaded dataset
# with this dataset, the app will create a supervised learning model with a user chosen target column with adjustable hyperparameters
# then produce two performance metrics on test data with explanations
# I am going to focus on Regression because that will be the most applicable to a variety of datasets

# intializing environment to hold all necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.markdown("## Select a Dataset: ")
st.text("You can use the example Social Media dataset, or upload your own")

toggle = st.toggle("I will use my own dataset")

if toggle:
    file = st.file_uploader("Upload your dataset of interest here", type="csv", key = 'user_upload')
else:
    file = "Social_media_impact_on_life.csv"

if file is not None:
    dataframe = pd.read_csv(file)
    st.dataframe(dataframe.head())
else:
    st.info("Please upload a CSV file to get started.")

st.markdown('## Now we will begin building the regression model!')
categories = dataframe.select_dtypes(include=['float', 'int']).columns.tolist()
target = st.selectbox('Which numerical value is your target feature?', categories)

X = pd.DataFrame(dataframe[categories].drop(target))
y = pd.Series(dataframe[target])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)