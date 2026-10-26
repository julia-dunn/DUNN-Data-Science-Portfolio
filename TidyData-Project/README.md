# Tidy Data Project: 2008 Olympic Medalists

## Project Overview
I created this Jupiter Notebook to further develop my skills in tidying data such that it can be analyzed and understood logically. 
This notebook's pivot tables and visualizations push beyond my first Tidy Data Project (Federal Research and Development Budgets) because the categorical data forces a level of creativity.
This is a cumulation not only of data tidying principles and methods, but also of how to analyze and visualize neat data represented in dataframes. 
The goal of this project was to organize an external dataset, which contained information regarding 2008 Olympic medalists for both men and women.

According to the Tidy Data Principles, neat data is such that:
- Every variable forms a column
- Every observation forms a row
- Each value is a cell

Prior to my analysis, the data was jumbled such that one column contained multiple variables. Here is a demonstration of how to melt data:
```
df_olymp_melted = pd.melt(df_olymp,
                          id_vars = 'medalist_name', # holds constant the column for medalist name
                          value_vars = df_olymp.columns[1:], # selects all the columns to be reoriented
                          var_name = 'sex_sport', # renames the column now containing the previous column names
                          value_name='Medal') #names the column containing all of the values from the original untidy dataframe
```
