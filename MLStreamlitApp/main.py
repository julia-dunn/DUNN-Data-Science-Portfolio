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
final_feat = []

with tab1:
    st.header("1. Select Your Dataset", divider = 'gray')
    st.caption("Choose a built-in example dataset or upload your own CSV file to begin.")

    with st.container(border = True):
        toggle = st.toggle("Upload my own dataset")

        if toggle:
            file = st.file_uploader("Upload your CSV file", type="csv", key = 'user_upload')
        else:
            file = st.radio('Choose an Example Dataset', ["Social_media_impact_on_life.csv", "titanic_data.csv"], horizontal = True)

    if file is not None:
        dataframe = pd.read_csv(file)
        cleaned_df = dataframe.dropna()
        
        st.subheader('Dataset Overview')

        m1, m2, m3 = st.columns(3)
        m1.metric('Total Rows', f'{len(dataframe):,}')
        m2.metric('Clean Rows (No NAs)', f'{len(cleaned_df):,}')
        m3.metric('Columns', len(dataframe.columns))

        with st.expander('Preview Raw Data'):
            st.dataframe(dataframe.head(), width='stretch')

        st.header('2. Configure your Model', divider = 'gray')

        with st.container(border=True):
            models = st.radio('Which type of regression model do you want to build', ['Linear Regression', 'Logistic Regression'], horizontal = True)
        
            num_feat = cleaned_df.select_dtypes(include=['float', 'int']).columns.tolist()
        
            if models == 'Linear Regression':
                target = st.selectbox('Select your numerical **target variable**:', num_feat, help = 'Tip: For the Social Media dataset, try selecting "Mental_Health_Index"')
                
                X = cleaned_df[num_feat].drop(columns=[target])
                x_list = X.columns.tolist()
                y = cleaned_df[target]

                st.markdown(f'#### Feature Relationshops vs. {target}')
                st.caption('Look for variables that show a clear linear trend')
                
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
        
            else:
                target = st.selectbox('Select your categorical **target variable**',cleaned_df.columns.tolist(), help='Tip: For the Titanic dataset, try selecting "survived"')
                if target: 
                    unique_vals = cleaned_df[target].unique()
                    st.markdown(f'**Unique values in {target}:** {unique_vals}')
                    log_condit = st.radio('Is this feature binary? (Yes/No, Pass/Fail, 0/1 etc.)',['Yes', 'No'], horizontal = True)
                    if log_condit == 'Yes':
                        st.success(f'**Target verified!** {target} is ready for Logistic Regression.\n\n'
                                   f"Switch to the **Building and Evaluating Model** tab to continue.")
                    else:
                        st.warning(f'**Invalid Target:** {target} has {len(unique_vals)} unique values.'
                                   f'Standard Logistic Regression requires a binary target.')

        if models == 'Linear Regression':
            st.header("3. Linearity Check", divider = 'gray')
            with st.container(border = True):
                st.markdown('Linear regresision works best when at least one feature demonstrates a linear relationship between the x values and the target feature')
                toggle1 = st.toggle('This dataset DOES show a linear relationship with the target')
                if toggle1:
                    final_feat = st.multiselect('Select the features to include in your model:', x_list)
                    if final_feat:
                        st.success(f"**Ready to train!**\n\n"
                        f"* **Target Variable:** {target}\n"
                        f"* **Selected Features:** {', '.join([f'{f}' for f in final_feat])}\n\n"
                        f"Switch to the **Building and Evaluating Model** tab above to run your model.")
                    else:
                        st.warning("Please select at least one feature above to continue.")
                else:
                    st.info("**Tip**: If the scatter plots look curved or groups into categories, try switching to **Logistic Regression** above.")
    else:
        st.info('Please upload a CSV file to get started.')
with tab2:
    st.header("1. Building the Basic Model", divider = "gray")

    if models == 'Linear Regression':
        st.caption('In linear regression models, we can choose to scale the data, such that the coefficients of the final model represent the effect of one standard deviation change in the feature. We can also choose to include an intercept B_0, as opposed to fitting through the origin.')
        X_selected = cleaned_df[final_feat]
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size = 0.2)
        lin_reg_unsc = LinearRegression()
        lin_reg_unsc.fit(X_train, y_train)

        y_pred_unsc = lin_reg_unsc.predict(X_test)

        mse_unsc = mean_squared_error(y_test, y_pred_unsc)
        rmse_unsc = root_mean_squared_error(y_test, y_pred_unsc)
        r2_unsc = r2_score(y_test, y_pred_unsc)

        st.subheader('Unscaled Model Coefficients')
        coef_df = pd.DataFrame({
                "Feature": final_feat,
                "Coefficient": lin_reg_unsc.coef_
            })
        st.dataframe(coef_df, width='stretch', hide_index=True)