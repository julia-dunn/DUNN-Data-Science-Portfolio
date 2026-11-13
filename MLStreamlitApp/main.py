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
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score

tab1, tab2, tab3 = st.tabs(['Selecting Dataset', 'Building and Evaluating Model', 'Diving Deeper'])

with tab1:
    st.markdown("## Select a Dataset: ")
    st.text("You can use the example Social Media dataset, example Titanic dataset, or upload your own")

    toggle = st.toggle("I will use my own dataset")

    if toggle:
        file = st.file_uploader("Upload your dataset of interest here", type="csv", key = 'user_upload')
    else:
        file = st.radio('Select an Example Dataset', ["Social_media_impact_on_life.csv", "titanic_data.csv"])

    if file is not None:
        dataframe = pd.read_csv(file)
        st.dataframe(dataframe.head())
        cleaned_df = dataframe.dropna()
        
        st.markdown('## Now we will build the regression model!')
        models = st.radio('What sort of regression model do you wish to use?', ['Linear Regression', 'Logistic Regression'])
        
        num_feat = cleaned_df.select_dtypes(include=['float', 'int']).columns.tolist()
        categories = cleaned_df.select_dtypes(include=['object', 'string']).columns.tolist()
        if models == 'Linear Regression':
            target = st.selectbox('Which numerical value is your target feature? For our dataset, we recommend Mental_Health_Index for our Social Media dataset', num_feat)
            X = cleaned_df[num_feat].drop(columns=[target])
            x_list = X.columns.tolist()
            y = cleaned_df[target]
            cols = st.columns(2)
    
            for idx, i in enumerate(x_list):
                fig, ax = plt.subplots(figsize=(5, 3.5))
                ax.scatter(X[i], y, alpha=0.6)
                ax.set_xlabel(i)
                ax.set_ylabel(target)
                ax.set_title(f"{i} vs. {target}")
    
                with cols[idx % 2]:
                    st.pyplot(fig)
                plt.close(fig)
    
            st.header('Is this data reasonable for linear regression?')
            st.text('If the data demonstrates a linear relationship between the x values and target feature, you can continue!')
            toggle1 = st.toggle('This dataset demonstrates a linear relationship between at least one x value and target feature, to create a linear regression. model')
            if toggle1: 
                final_feat = st.multiselect('Which features demonstrate linear relationships, and can thus be used for the linear regression model?', x_list)
                st.header('So what is our model doing?')
                st.text(f'Our target variable: {target}')
                st.text(f'The features used to predict the target: {final_feat}')
                st.text('Navigate to the "Building and Evaluating Model" tab to create a linear regression model and see how well it works!')
            else:
                st.write("If this dataset doesn't seem reasonable for linear regression, try logistic regression!")
    
        else:
            target = st.selectbox('Which feature is your target feature? We recommend the "survived" feature.',cleaned_df.columns.tolist())
            if target: 
                unique_vals = cleaned_df[target].unique()
                st.write(f'Unique values in {target}')
                st.write(unique_vals.tolist())
                log_condit = st.radio('Is this feature binary? (Yes/No, Pass/Fail, 0/1 etc.)',['Yes', 'No'])
                if log_condit == 'Yes':
                    st.write('Amazing! We can build a logisitic regression model')
                    st.text('Navigate to the "Building and Evaluating Model" tab to create a logistic regression model and see how well it works!')
                else:
                    st.write('Oh no! This target feature cannot be used for logistic regression!')
    else:
        st.info("Please upload a CSV file to get started.")

with tab2:
    st.markdown("## Let's build our model!")
    