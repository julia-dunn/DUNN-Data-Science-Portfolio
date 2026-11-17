# Linear and Logistic Regression: What works for *your* dataset?
Find this app [here!](https://linlogregressiondunn.streamlit.app/)
## Overview
This app can be used to create both linear and logistic regression models of datasets with primarily numerical variables. 
It walks the user through the process of cleaning data (on a low level), selecting the type of regression that makes sense, and choosing variables and target features that fit the requirements of the specified model. 
Most importantly, the user is pushed to think critically about *if* the models can be applied to their data. It is entirely possible for beginner data analysists, such as sutdents like myself, to act based on assumptions on their data that are deeply flawed.
This app thinks deeply about if and when those assumptions can be applied, and then demonstrates the result of hyper-parameter fine-tuning once the assumptions have been made. 

## How to run locally
First, ensure that the file structure is correct such that the example datasets can be accessed. Your local file structure should remain as follows:
```
MLStreamlitApp
├──main.py
├──Social_media_impact_on_life.csv
├──titanic_data.csv
├──requirements.txt
```
Once your files are correctly organized, ensure you have the appropriate notebooks and lirbaries installed. These are described in requirements.txt, but I will repeat them here. 
```
matplotlib==3.11.2
numpy==2.5.5
pandas==3.1.1
scikit_learn==1.9.1
seaborn==0.13.2
statsmodels==0.15.0
streamlit==1.63.0
```
To install:
```
!pip install matplotlib
!pip install numpy
!pip install scikit_learn
!pip install seaborn
!pip install statsmodels
!pip install streamlit
```

To import (already at the top of main.py script):
```
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as pls
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
from statsmodels.graphics.gofplots import qqplot

from pathlib import Path
```

Once the requirements are installed and imported, the user may then create an integrated terminal inside of the MLStreamlitApp folder and simply run the command:
```
streamlit run main.py
```

## Models and their Hyperparameters

### Linear Regression
#### Metrics
- Root Mean Squared Error: The root mean squared error described how much the model's predictions differ from the 'actual' value of the target variable. It takes the sum of the difference between each value and its expected value, then averages that for the number of observations n, then takes the square root.
- R2 Score: The R2 score (aka the coefficient of determination) essentially describes the extent to which the variance in target variable can be explained by the predictor variables. It is compares the variance of the prediction error to the variance in the dataset as a whole.

#### Hyperparameters
- Scaling (Yes/No)
- Y-intercept (Yes/No)

### Logistic Regression

#### Metrics
- Accuracy: Accuracy represents the percentage of classifications that are correct, including both True Positive and True Negatives, with respect to the toal number of classifications.
- Precision: Precision indicates the number of correct positive classifications out of the total number of positive classifications. This is necessary if false positives are relevant in your domain.
- Recall: Recall indicates the number of correct positive classifications out of the total number of true positive datapoints. This indicates when positives are missing.
- F1 Score: The F1 score balances information regarding precision and recall. It is what is called a "balanced mean", two times the product divded by the sum.

#### Hyperparameters: 
- Max Number of Iterations (between 0 and 1000)
- Class Weight (None or Balanced)

## References 
- [Pathlib documentation](https://docs.python.org/3/library/pathlib.html)
- [Cache_Data Documentation](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_data)
- [Guide to Markdown Syntax for Formatting Streamlit App - citing entire page](https://isom3400-app.streamlit.app/text_elements)
- [StandardScaler Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

## [Working Through App with Examples (Video)](https://drive.google.com/file/d/1EMR1NZQsKnvHrcITY3Q4eQtGBejyprCs/view?usp=sharing)
