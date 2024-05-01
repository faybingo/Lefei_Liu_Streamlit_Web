import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
import os
import requests
import json
import base64
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import googleapiclient.discovery
import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import io
client_id='ec31934c1e8f48a788084e4210a2d355'
client_secret='ab60d2175d7347e79df96deffb125f6a'

# Function to load data
def load_data():
    try:
        # Assuming the script is being run from the directory where 'total_data.csv' is located
        # If that's not the case, you might need to adjust the 'file_path'
        file_path = 'total_data.csv'
        df_spo = pd.read_csv(file_path)
        return df_spo
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None
    

#get token
def get_token():
    client_id='ec31934c1e8f48a788084e4210a2d355'
    client_secret='ab60d2175d7347e79df96deffb125f6a'
    auth_string = client_id + ":" + client_secret
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")
                    
    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": "Basic " + auth_base64,
        "Content-Type": "application/x-www-form-urlencoded"
    }
                    
    data = {"grant_type": "client_credentials"}
    result = requests.post(url, headers=headers, data=data)
    json_result = result.json()
    token = json_result["access_token"]
    return token

def get_auth_header(token):
    client_id='ec31934c1e8f48a788084e4210a2d355'
    client_secret='ab60d2175d7347e79df96deffb125f6a'
    return {"Authorization": "Bearer " + token}

#get artists' id
def search_for_artist(token, artist_name):
    url = "https://api.spotify.com/v1/search"
    headers = get_auth_header(token)
    query = f"?q={artist_name}&type=artist&limit=1"
                    
    query_url = url + query
    result = requests.get(query_url, headers=headers)
    json_result = result.json()["artists"]["items"]
                    
    if len(json_result) == 0:
        print("No artist with this name exists.")
        return None
                    
    return json_result[0]

#get artists' songs according to their id
def get_songs_by_artist(token, artist_id):
    url = f"https://api.spotify.com/v1/artists/{artist_id}/top-tracks?country=US"
    headers = get_auth_header(token)
    result = requests.get(url, headers=headers)
    json_result = result.json()["tracks"]
    return json_result

# Define a function for the music analysis page
def page_music_analysis():
    df_spo = load_data()
    if df_spo is not None:
        st.title("Music Data Analysis")

        # Button to show data cleaning information
        if st.button('Show Data Cleaning Information'):
            with st.expander("Data Cleaning Information"):
                st.write("Missing values in each column:")
                st.write(df_spo.isnull().sum())
                duplicate_count = df_spo.duplicated().sum()
                st.write(f"Number of duplicate rows: {duplicate_count}")

        # Button to show data description
        if st.button('Show Data Description'):
            with st.expander("Data Description"):
                buffer = io.StringIO()
                df_spo.info(buf=buffer)
                s = buffer.getvalue()
                st.text(s)
                st.write("Summary statistics:")
                st.write(df_spo.describe())
                st.write("Count of songs per artist:")
                st.write(df_spo['artist_name'].value_counts())

        # Correlation heatmap
        if st.button('Show Correlation of Each Variable'):
            with st.expander("Correlation Heatmap"):
                def plot_correlation(data):
                    numeric_df = data.select_dtypes(include=[np.number])
                    corr_df = numeric_df.corr()
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.heatmap(corr_df, annot=True, cmap='coolwarm', ax=ax)
                    ax.set_title('Correlation between each variable')
                    st.pyplot(fig)
                plot_correlation(df_spo)

        if st.button('Show Histgram of Popularity'):
            with st.expander("Histgram of Popularity"):
                def plot_popularity_distribution(data):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(df_spo['popularity'], bins=20, color='lightblue', edgecolor='black')
                    ax.set_xlabel('Popularity')
                    ax.set_ylabel('Numbers')
                    ax.set_title('Distribution of popularity')
                    st.pyplot(fig)
                plot_popularity_distribution(df_spo)

        if st.button('Show Most Popular Songs'):
            with st.expander("Most Popular Songs"):
                def most_popular_songs(data):
                    songs = data[data['popularity'] > 90]
                    most_popular_songs_df = songs.sort_values('popularity', ascending=False).head(10)  # to see first 10 songs
                    return most_popular_songs_df

                def plot_most_popular_songs(most_popular_songs_df):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.countplot(data=most_popular_songs_df, x='popularity', palette='viridis', ax=ax)
                    ax.set_title('Most popular songs')
                    for bar in ax.containers:
                        ax.bar_label(bar)
                    ax.set_xlabel('Popularity')
                    ax.set_ylabel('Number of songs')
                    st.pyplot(fig)  # Display the figure with Streamlit

                popular_songs_df = most_popular_songs(df_spo)
                plot_most_popular_songs(popular_songs_df)
                st.write(popular_songs_df)
                st.write("We can see top 10 songs' artists are Ariana Grande,Post Malone and Daddy Yankee. So I will extract their top 10 songs from spotify api, and do later analysis based on these three artists.")

        if st.button('Show Top 10 Songs of above three artists'):
            with st.expander("Top 10 Songs"):
#Get top 10 songs of (Ariana Grande, Post Malone, Daddy Yankee) from api_scraper 
#code is the same with api_scraper.py
                

                #get token header
                
            
                


        

# Define a function for the Q&A page
def page_answer_questions():
    st.title("Answer the Questions")
    # You can use st.write or st.markdown to format your text
    st.markdown("""
        4. **What did you set out to study?**  
           I set out to study...

        5. **What did you discover/what were your conclusions?**  
           The findings were...

        6. **What difficulties did you have in completing the project?**  
           Some of the challenges...

        7. **What skills did you wish you had while you were doing the project?**  
           I wished to have skills like...

        8. **What would you do "next" to expand or augment the project?**  
           To expand the project...
    """)


def page_top_songs():
    st.title("Top 10 Songs of Artists")

    client_id = st.secrets["SPOTIFY_CLIENT_ID"]
    client_secret = st.secrets["SPOTIFY_CLIENT_SECRET"]
    token = get_token(client_id, client_secret)

    artist_names = ["Ariana Grande", "Post Malone", "Daddy Yankee"]
    top_songs = {}

    for artist_name in artist_names:
        artist = search_for_artist(token, artist_name)
        if artist:
            artist_id = artist["id"]
            songs = get_songs_by_artist(token, artist_id)
            top_songs[artist_name] = songs[:10]  # Get the top 10 songs

    # Display the top songs for each artist
    for artist_name, songs in top_songs.items():
        st.subheader(f"Top songs by {artist_name}:")
        for idx, song in enumerate(songs):
            st.write(f"{idx+1}. {song['name']} - Popularity: {song['popularity']}")

# Streamlit app logic
page = st.sidebar.selectbox("Choose a page", ["Music Analysis", "Answer the Questions", "Top Songs"])

if page == "Music Analysis":
    page_music_analysis()
elif page == "Answer the Questions":
    page_answer_questions()
elif page == "Top Songs":
    page_top_songs()
