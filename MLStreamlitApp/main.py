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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix


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
                    num_feat = cleaned_df.select_dtypes(include=['float', 'int']).columns.tolist()

                    if log_condit == 'Yes':
                        st.success(f'**Target verified!**')
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
            st.header("3. Encoding Categorical Variables", divider = 'gray')
            with st.container(border = True):
                feat_encode = st.multiselect('Select which features are categorical, but can be numerically encoded',cleaned_df.drop(columns = [target]).columns.tolist())
                selected_feat = list(dict.fromkeys(num_feat + feat_encode))
                encoded_df = cleaned_df[selected_feat]
                encoding_maps = {}
                for feat in feat_encode:
                    st.markdown(f'**Encoding for {feat}:**')
                    unique_val_feats = cleaned_df[feat].unique()

                    col_map = {}
                    cols = st.columns(len(unique_val_feats))
                    for idx, val in enumerate(unique_val_feats):
                        with cols[idx]:
                            col_map[val] = st.number_input(f"Value for {val}", value= idx, step = 1, key = f"encode_{feat}_{val}")

                    encoding_maps[feat] = col_map
                    encoded_df[feat] = encoded_df[feat].map(encoding_maps[feat])

                X = encoded_df.drop(columns=[target], errors = 'ignore')
                y = encoded_df[target]

                st.success(f"**Ready to train!**\n\n"
                           f"* **Target Variable:** {target}\n"
                           f"* **Selected Features:** {', '.join([f'{f}' for f in selected_feat])}\n\n"
                           f"Switch to the **Building and Evaluating Model** tab above to run your model.")

    else:
        st.info('Please upload a CSV file to get started.')
with tab2:
    st.header("1. Building the Basic Model", divider = "gray")
    if file is None:
        st.info("Please select or upload a dataset in the **Selecting Dataset** tab first.")
    elif models == 'Linear Regression':
        if len(final_feat) == 0:
            st.warning("No features have been selected yet. Please navigate to the **Selecting Dataset** tab and confirm a linear relationship.")
        else:
            with st.container(border = True):
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
                coef_df_unsc = pd.DataFrame({
                        "Feature": final_feat,
                        "Coefficient": lin_reg_unsc.coef_
                    })
                st.dataframe(coef_df_unsc, width='stretch', hide_index=True)

                st.subheader('Unscaled Fit Metrics')
                m1, m2, m3 = st.columns(3)
                m1.metric('Mean Squared Error', f'{mse_unsc:.4f}')
                m2.metric('Root Mean Squared Error', f'{rmse_unsc:.4f}')
                m3.metric('R2 Score', f'{r2_unsc:.4f}')
    else:
        if target:
            with st.container(border=True):
                st.caption('In Logistic Regression, we can select for a few different types of hyperparameters. Primarily, we will focus on class weight and the maximum number of iterations. First we will build the default model.')
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                log_reg_default = LogisticRegression()
                log_reg_default.fit(X_train, y_train)

                y_pred = log_reg_default.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                st.subheader('Default Fit Metrics')
                m1, m2, m3, m4 = st.columns(4)

                m1.metric('Accuracy', f'{accuracy:.4f}')
                m2.metric('Precision', f'{precision:.4f}')
                m3.metric('Recall', f'{recall:.4f}')
                m4.metric('F1', f'{f1:.4f}')

                st.subheader('Confusion Matrix for Default Model')

                cm1 = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5,4))
                sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)


        else:
            st.warning('No target has been selected. Please navigate to the **Selecting Dataset**.')
            
    if models == 'Linear Regression':
        st.header('2. Linear Regression Hyperparameters', divider = 'gray')
        with st.container(border = True):
            st.markdown('For your dataset, select if you want to scale the data (making the coefficients more readable), and if the intercept assumption should be eliminated (dependent on domain knowledge)')
            toggle2 = st.toggle('I want to scale the data')
            toggle3 = st.toggle('I want to fit through the origin')

            if toggle2: 
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                learned = pd.DataFrame({
                    "mu (training mean)": scaler.mean_,
                    "sigma (training std)": scaler.scale_,
                    }, index=final_feat)

                st.dataframe(learned, width='stretch', hide_index = True)

                if toggle3:
                    lin_reg_scaled_origin = LinearRegression(fit_intercept = False)
                    lin_reg_scaled_origin.fit(X_train_scaled, y_train)
                    y_pred_scaled_origin = lin_reg_scaled_origin.predict(X_test_scaled)

                    mse_scaled_origin  = mean_squared_error(y_test, y_pred_scaled_origin)
                    rmse_scaled_origin = root_mean_squared_error(y_test, y_pred_scaled_origin)
                    r2_scaled_origin   = r2_score(y_test, y_pred_scaled_origin)

                    st.subheader('Scaled Model Coefficients')
                    coef_df_sc_o = pd.DataFrame({
                            "Feature": final_feat,
                            "Coefficient": lin_reg_scaled_origin.coef_
                        })
                    st.dataframe(coef_df_sc_o, width='stretch', hide_index=True)
        
                    st.subheader('Scaled Fit Metrics')
                    m1, m2, m3 = st.columns(3)
                    m1.metric('Mean Squared Error', f'{mse_scaled_origin:.4f}')
                    m2.metric('Root Mean Squared Error', f'{rmse_scaled_origin:.4f}')
                    m3.metric('R2 Score', f'{r2_scaled_origin:.4f}')

                else:
                    lin_reg_scaled = LinearRegression()
                    lin_reg_scaled.fit(X_train_scaled, y_train)
                    y_pred_scaled = lin_reg_scaled.predict(X_test_scaled)
                    
                    mse_scaled  = mean_squared_error(y_test, y_pred_scaled)
                    rmse_scaled = root_mean_squared_error(y_test, y_pred_scaled)
                    r2_scaled   = r2_score(y_test, y_pred_scaled)

                    st.subheader('Scaled Model Coefficients')
                    coef_df_sc = pd.DataFrame({
                            "Feature": final_feat,
                            "Coefficient": lin_reg_scaled.coef_
                        })
                    st.dataframe(coef_df_sc, width='stretch', hide_index=True)
        
                    st.subheader('Scaled Fit Metrics')
                    m1, m2, m3 = st.columns(3)
                    m1.metric('Mean Squared Error', f'{mse_scaled:.4f}')
                    m2.metric('Root Mean Squared Error', f'{rmse_scaled:.4f}')
                    m3.metric('R2 Score', f'{r2_scaled:.4f}')
    else:
        st.header('Logistic Regression Hyperparameters', divider = 'gray')