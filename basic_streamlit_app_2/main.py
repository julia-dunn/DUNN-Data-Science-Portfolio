# intializing environment to hold all necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# loading the spotify dataset

# loading the spotify dataset
df = pd.read_csv("data/spotify_dataset.csv")

# creating heading of app with logo on the left
logo, title = st.columns([1, 8], vertical_alignment="center")

with logo:
    st.image("https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg", width=65)
with title:
    st.title("What is there to listen to on Spotify?")

st.markdown("Use this app to learn about a huge variety of songs on spotify. Ask questions like 'What genre is the most dancable?' and 'What is the relationship between tempo and valence?' This is your chance to dive deep into the statistics behind your favorite songs. Enjoy! ")

tab1, tab2, tab3, tab4 = st.tabs(['Home', 'Danceability', 'Explicitness', 'Key'])

with tab1:

    st.write('### Summary Statistics of Spotify Dataset:')
    st.dataframe(df.describe())
    # the summary statistics of the dataset demonstrates that there is no missing data, because all of the counts are the same number, 114,000

    st.markdown("### Let's Explore!")
    st.markdown("Use these selectors and sliders to learn about different trends. What do you notice?")
    # select a genre to filter the dataset 

    genre = st.selectbox('Genre',df['track_genre'].unique())
    df_filtered = df[df['track_genre'] == genre]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Full Data Distrubution')
        # plot a histogram with KDE for selected column from original database (kernal density estimation)
        fig1, ax1 = plt.subplots(figsize=(5, 3.5))
        sns.histplot(df['popularity'], kde=True, ax=ax1)
        plt.title(f'Original Distribution of')
        st.pyplot(fig1)
        st.subheader(f"Original Status")
        # display statistical summary for the selected column
        st.write(df['popularity'].describe())

    # Cleaned Data Visualization
    with col2:
        st.subheader(f'{genre} Data Distrubution')
        # plot a histogram with KDE for selected column from original database (kernal density estimation)
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        sns.histplot(df_filtered['popularity'], kde=True, ax=ax2)
        plt.title(f'Cleaned Distribution of {genre}')
        st.pyplot(fig2)
        st.subheader(f"{genre}'s Cleaned Status")
        # display statistical summary for the selected column
        st.write(df_filtered['popularity'].describe())

    min_duration = df['duration_ms'].min()
    max_duration = df['duration_ms'].max()

    duration_range = st.slider(
        'Song Duration (ms)',
        min_value = min_duration,
        max_value = max_duration,
        value = (min_duration,max_duration),
        step = 1000
    )

    df_filtered_2 = df[(df['duration_ms'] >= duration_range[0]) 
                    & (df['duration_ms'] <= duration_range[1])
                    ]

    col3, col4 = st.columns(2)

    with col3:
        st.subheader('Full Data Distrubution')
        # plot a histogram with KDE for selected column from original database (kernal density estimation)
        fig1, ax1 = plt.subplots(figsize=(5, 3.5))
        sns.histplot(df['danceability'], kde=True, ax=ax1)
        plt.title(f'Original Distribution of')
        st.pyplot(fig1)
        st.subheader(f"Original Status")
        # display statistical summary for the selected column
        st.write(df['danceability'].describe())

    # Cleaned Data Visualization
    with col4:
        st.subheader('Data Distrubution within range')
        # plot a histogram with KDE for selected column from original database (kernal density estimation)
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        sns.histplot(df_filtered_2['danceability'], kde=True, ax=ax2)
        plt.title(f'Data Distribution within Range')
        st.pyplot(fig2)
        st.subheader(f"Status within Range")
        # display statistical summary for the selected column
        st.write(df_filtered_2['danceability'].describe())