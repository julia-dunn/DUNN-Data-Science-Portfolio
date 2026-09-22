<div align="left">
  <img src="https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg" width="60" align="left" style="margin-right: 15px;">
  <h1>What is there to listen to on Spotify?</h1>
</div>

### Project Overview 
Use this app to learn about a huge variety of songs on spotify. Ask questions like 'What genre is the most dancable?' and 'What is the relationship between tempo and valence?' This is your chance to dive deep into the statistics behind your favorite songs. Enjoy!

This app is essentially an exploratory data analytics dashboard, where you can learn about important relationships between numerical and categorical variables. Here, we explore a dataset of over 100,000 songs on spotify. Check out this user-interactive platform and generate visualizations yourself!

#### What's in the dataset:
This dataset features over 100,000 songs, each of which have the following features and variables:
- Track ID
- Artists
- Track name
- Popularity (value from 0-100)
- Duration (ms)
- Danceability (0-1)
- Energy (0-1)
- Key (numeber correlates the note)
- Loudness (dB)
- Mode
- Speechiness (0-1)
- Acousticness (0-1)
- Instrumetnalness (0-1)
- Liveness (0-1)
- Valence (0-1)
- Tempo (bpm)
- Time signature
- Track genre

### Setup and Run Instructions: 
There are a few required libraries for this app in order to run the necessary calculations. They are listed in the requirements file within the portfolio, and include:

- Streamlit 1.63.0
- Pandas 2.2.2
- Matpltlib 3.9.2
- Seaborn 0.13.2

To install these libraries, run the following commands in your terminal: 

```
{
    pip install streamlit
    pip install pandas
    pip install matplotlib
    pip install seaborn
}
```

Then, to run the app locally, run this command in the integrated terminal.

```
{
    steamlit run main.py
}
```
 