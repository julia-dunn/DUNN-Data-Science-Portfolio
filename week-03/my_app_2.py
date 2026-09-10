import streamlit as st
import pandas as pd

st.title('Hello Streamlit') # this is the same as one # in markdown syntax
st.write("This is my first Streamlit app.")

if st.button('Click me!'):
    st.write('🎉 You clicked the button! Nice work! 🚀')
else:
    st.write("Click the button to see what happens...")

color = st.color_picker('Pick a color', '#00f900')
st.write(f"You picked: {color}")

st.subheader("Exploring our Dataset")

# load the CSV file

df = pd.read_csv('data_3/sample_data-1.csv')

st.write("Here's our data:")
st.dataframe(df)

# Filter

city = st.selectbox("select a city", df['City'].unique())
filtered_df = df[df['City'] == city]

st.write(f"People in {city}:")
st.dataframe(filtered_df)