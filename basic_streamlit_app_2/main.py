# intializing environment to hold all necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

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

tab1, tab2, tab3, tab4 = st.tabs(['Home', 'Artist', 'Danceability', 'Key'])

with tab1:

    st.write('### Summary Statistics of Spotify Dataset:')
    st.dataframe(df.describe())
    # the summary statistics of the dataset demonstrates that there is no missing data, because all of the counts are the same number, 114,000

    st.markdown("### Let's Explore!")
    st.markdown("Use these selectors and sliders to learn about different trends. What do you notice?")
    # select a genre to filter the dataset 
    st.markdown('###### Here you can see how the genre distribution changes for individual genres.')
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

    # Filtered Genre Data Visualization
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

    st.markdown('###### Here you can compare these different numarical metrics and their trends across different keys')
    # filtered by key
    keys_conversion = { # in dataset, key is indicated by number (not intuitive) this is a dictionary to map the number to the key letter
        0:'C', 1:'C-s/D-f', 2:'D', 3:'D-s/E-f', 4:'E', 5: 'F', 
        6:'F-s/G-d', 7:'G', 8: 'G-s/A-f', 9:'A', 10:'A-s/B-f', 11:'B'
    }
    # add dataframe column maping key names to numbers
    df['key_name'] = df['key'].map(keys_conversion)
    #put keys in order of the actual ascending letters
    key_order = [keys_conversion[i] for i in range(11)]

    unique_keys = list(sorted(df['key'].unique()))

    selected_key = st.radio(
        'Numerical Comparison',
        options = ['speechiness', 'energy', 'acousticness']
    )
    # create a boxplot plot of the different numerical values by key
    fig3, ax3 = plt.subplots()    
    sns.boxplot(data=df, x = selected_key, y = 'key_name',order = key_order, ax = ax3, palette = 'muted')
    plt.title(f'{selected_key} vs Key')
    st.pyplot(fig3)

    st.write("To learn more about how the trends in data relate to the key the song is written in, click the 'Key' tab!")

    # filtered by duration
    min_duration = df['duration_ms'].min()
    max_duration = df['duration_ms'].max()

    duration_range = st.slider(
        'Song Duration (ms)',
        min_value = min_duration,
        max_value = max_duration,
        value = (min_duration,max_duration),
        step = 1000
    )
    # creates a dataframe filtered by the duration range set by the user
    df_filtered_2 = df[(df['duration_ms'] >= duration_range[0]) 
                    & (df['duration_ms'] <= duration_range[1])
                    ]

    col3, col4 = st.columns(2)
    #plots the original distribution and the distribution of data for songs within range, easy to compare the two along with their summary statistics 
    with col3:
        st.subheader('Full Data Distrubution')
        # plot a histogram with KDE for selected column from original database (kernal density estimation)
        fig4, ax4 = plt.subplots(figsize=(5, 3.5))
        sns.histplot(df['danceability'], kde=True, ax=ax4)
        plt.title(f'Original Distribution of')
        st.pyplot(fig4)
        st.subheader(f"Original Status")
        # display statistical summary for the selected column
        st.write(df['danceability'].describe())

    # Cleaned Data Visualization
    with col4:
        st.subheader('Data Distrubution within range')
        # plot a histogram with KDE for selected column from original database (kernal density estimation)
        fig5, ax5 = plt.subplots(figsize=(5, 3.5))
        sns.histplot(df_filtered_2['danceability'], kde=True, ax=ax5)
        plt.title(f'Data Distribution within Range')
        st.pyplot(fig5)
        st.subheader(f"Status within Range")
        # display statistical summary for the selected column
        st.write(df_filtered_2['danceability'].describe())

    # using tabs for easy user navigation to specific sub topics
    with tab2:
        st.write('Here, you can search for a specific artist! See how they compare to the entire dataset with respect to popularity, danceability, energy, explicitness, loudness, speechiness, acousticness, instrumentalness, liveness, and valence.')

        search = st.text_input("What artist do you want to learn more about?", placeholder= 'Search for artist here')

        if search:
            filtered_df_4 = df[df['artists'].str.contains(search, case=False, na=False)]
            st.markdown(f"#### Songs by {search}")
            st.dataframe(filtered_df_4)

            numerical_categories = [
                'popularity', 'danceability', 'energy', 'loudness',
                'speechiness', 'acousticness',
                'liveness', 'valence', 'tempo', 'duration_ms'
            ]
            st.markdown(f"### {search} Compared to Entire Dataset")
            fig, axes = plt.subplots(5,2, figsize=(8, 20))
            axes = axes.flatten()
            for i, category in enumerate(numerical_categories):
                ax = axes[i]
                sns.histplot(
                    df[category], bins=30, stat = 'density', color = 'gray', edgecolor='black', # using density stat to compare shapes of plots, as absolute counts make artist insignificant
                    alpha = 0.35, label = 'All Spotify Songs', ax=ax
                )
                sns.histplot(
                    filtered_df_4[category], bins = 30, stat = 'density', color = 'green', edgecolor='none', 
                    alpha = 0.45, label = search, ax = ax
                )
                ax.set_title(f"{category} Distribution")
                ax.set_xlabel(f"{category}")
                ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
        else: 
            st.markdown('### Search for data visualizations to appear!')

    with tab3: 
        st.write('Here you can learn about what songs are the most (or least) danceable! Features we have considered here include genre, key, loudness, valence, and tempo')

        # groups danceability by genre for visualization of means

        st.write('Select genres to compare:')

        genres=df['track_genre'].unique()
        selected_genres = []

        columns = st.columns(7)
        for i, genre in enumerate(genres):
            column = columns[i % 7]
            with column:
                if st.checkbox(genre):
                    selected_genres.append(genre)

        df_filtered_5 = df[df['track_genre'].isin(selected_genres)] 
        genre_danceability = (df_filtered_5.groupby('track_genre')['danceability'].mean())

        fig, ax= plt.subplots()
        plt.bar(genre_danceability.index, genre_danceability.values)
        plt.xlabel("Genre")
        plt.ylabel('Average Danceability')
        plt.title('Average Danceability by Genre')

        st.pyplot(fig)

        keys_conversion = { # in dataset, key is indicated by number (not intuitive) this is a dictionary to map the number to the key letter
            0:'C', 1:'C-s/D-f', 2:'D', 3:'D-s/E-f', 4:'E', 5: 'F', 
            6:'F-s/G-d', 7:'G', 8: 'G-s/A-f', 9:'A', 10:'A-s/B-f', 11:'B'
        }
            # add dataframe column maping key names to numbers
        df['key_name'] = df['key'].map(keys_conversion)
        key_danceability = (df.groupby('key_name')['danceability'].mean())

        fig, ax= plt.subplots()
        plt.bar(key_danceability.index, key_danceability.values)
        plt.xlabel("Key")
        plt.ylabel('Average Danceability')
        plt.title('Average Danceability by Key')
        
        st.pyplot(fig)
        
        fig, ax = plt.subplots()

        plt.scatter(
            df['loudness'],
            df['danceability']
        )
        plt.xlabel('Loudness')
        plt.ylabel('Danceability')
        plt.title('Danceability vs Loudness')

        # axes[1].violinplot(
        #     df['valence'],
        #     df['danceability']
        # )
        # axes[1].set_xlabel('Valence')
        # axes[1].set_ylabel('Danceability')
        # axes[1].set_title('Danceability vs Valence')

        # axes[2].violinplot(
        #     df['tempo'],
        #     df['danceability']
        # )
        # axes[2].set_xlabel('Tempo')
        # axes[2].set_ylabel('Danceability')
        # axes[2].set_title('Danceability vs Tempo')

        plt.tight_layout()

        st.pyplot(fig)