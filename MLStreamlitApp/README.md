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
