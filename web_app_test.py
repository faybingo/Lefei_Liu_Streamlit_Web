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
import requests
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import os

client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

print("Client ID:", client_id)
print("Client Secret:", client_secret)

# Spotify API Functions
def get_token(client_id, client_secret):
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    
    auth_url = "https://accounts.spotify.com/api/token"
    auth_response = requests.post(auth_url, {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret,
    })
    if auth_response.status_code == 200:
        return auth_response.json()['access_token']
    else:
        st.error("Failed to retrieve token")
        return None

def search_for_artist(token, artist_name):
    search_url = f"https://api.spotify.com/v1/search?q={artist_name}&type=artist&limit=1"
    response = requests.get(search_url, headers={"Authorization": f"Bearer {token}"})
    if response.status_code == 200 and response.json()['artists']['items']:
        return response.json()['artists']['items'][0]['id']
    else:
        st.error(f"No artist found with the name {artist_name}")
        return None

def get_songs_by_artist(token, artist_id):
    tracks_url = f"https://api.spotify.com/v1/artists/{artist_id}/top-tracks?market=US"
    response = requests.get(tracks_url, headers={"Authorization": f"Bearer {token}"})
    if response.status_code == 200:
        return response.json()['tracks']
    else:
        st.error("Failed to retrieve songs")
        return []

# Load data
def load_data():
    try:
        df_spo = pd.read_csv('total_data.csv')
        return df_spo
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

# Music Analysis Page
def page_music_analysis():
    df_spo = load_data()
    if df_spo is not None:
        st.title("Music Data Analysis")
        st.markdown("""
            **Developed by: Lefei Liu

            ### How to Use This Web
            - **Interactivity:** You can interact with the app by clicking on buttons like 'Show Data Cleaning Information', 'Show Correlation of Each Variable', etc. Each button will provide detailed insights or visualizations based on the underlying music data.
            - **Charts and Plots:** The charts represent various analytical insights such as the correlation between musical features, popularity distributions, and more.
            - **Conclusions:** After exploring the data, the app provides conclusions based on the analysis such as the most popular artists, the characteristics of popular music, etc.

            ### Major Gotchas
            - **Performance:** Random Forest may cost 5 minutes to run.
            - **Improvements:** The application can be further improved by integrating real-time data and using more advanced machine learning models for predictions.
        """)

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
                st.write("We can see that the top 10 songs' artists are Ariana Grande, Post Malone, and Daddy Yankee. Therefore, I will extract their top 10 songs from the Spotify API (under the 'Top Songs' page) and conduct further analysis based on these three artists.")

        
        if st.button('Show Sales Amounts and Compare'):
            with st.expander("Sales Amount of Ariana Grande"):
                script_path = os.path.dirname(__file__)
                filename_ag = 'amazon_ag'
                file_path_ag = os.path.join(script_path, f'{filename_ag}.csv')
                df_ag = pd.read_csv(file_path_ag)
                keys = df_ag.keys()
                col1_name = keys[0]
                col2_name = keys[1]
                df_ag = df_ag.rename(columns={col1_name: 'sales amount', col2_name: 'price'})
                df_ag['sales amount'] = df_ag['sales amount'].str.replace(r'1K+', '1000+')
                df_ag['sales amount'] = df_ag['sales amount'].str.replace(r'\D+', '', regex=True)
                df_ag['sales amount'] = df_ag['sales amount'].astype(int)
                st.write("Ariana Grande's sales amount:")
                st.write(df_ag)
                sales_ag = df_ag['sales amount'].sum()
                st.write("Ariana Grande's total sales amount:", sales_ag)

            with st.expander("Sales Amount of Post Malone"):
                filename_pm = 'amazon_pm'
                file_path_pm = os.path.join(script_path, f'{filename_pm}.csv')
                df_pm = pd.read_csv(file_path_pm)
                keys = df_pm.keys()
                col1_name = keys[0]
                col2_name = keys[1]
                df_pm = df_pm.rename(columns={col1_name: 'sales amount', col2_name: 'price'})
                df_pm['sales amount'] = df_pm['sales amount'].str.replace(r'\D+', '', regex=True)
                df_pm['sales amount'] = df_pm['sales amount'].astype(int)
                st.write("Post Malone's sales amount:")
                st.write(df_pm)
                sales_pm = df_pm['sales amount'].sum()
                st.write("Post Malone's total sales amount:", sales_pm)

            # Comparison and Display
            df_compare = pd.DataFrame({
                'Artist': ['Ariana Grande', 'Post Malone'],
                'Total Sales': [sales_ag, sales_pm]
            })
            st.write("Comparison of Total Sales:")
            st.write(df_compare)

            explanation = """
            This comparison shows the sales amount of Ariana Grande and Post Malone.
            Note: Daddy Yankee's monthly sales data below 50 are not displayed on Amazon.
            """
            st.write(explanation)


        def generate_and_display_wordcloud(file_path, artist_name):
            try:
                df = pd.read_csv(file_path)
                df['text'] = df['text'].str.replace('Evan Peter', '') 
                comments = df['text'].astype(str).tolist()
                comments_text = ' '.join(comments)
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(comments_text)

                fig, ax = plt.subplots(figsize=(10, 8))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Failed to load or generate the wordcloud for {artist_name}: {e}")

        if st.button('Show Wordclouds'):
            artists = {
                'Ariana Grande': 'youtube_ag.csv',
                'Post Malone': 'youtube_pm.csv',
                'Daddy Yankee': 'youtube_dy.csv'
            }
            script_path = os.path.dirname(__file__)

            for artist_name, filename in artists.items():
                with st.expander(f"Wordcloud of {artist_name}"):
                    file_path = os.path.join(script_path, filename)
                    generate_and_display_wordcloud(file_path, artist_name)

        def plot_average_popularity_by_genre(data):
            # Grouping data by genre and calculating mean popularity
            gen = (data.groupby('genre')['popularity']
                .mean()
                .sort_values(ascending=False)
                .reset_index()
                .rename(columns={'popularity': 'Average popularity'}))
            
            # Plotting
            plt.figure(figsize=(10, 6))
            plt.bar(gen['genre'], gen['Average popularity'], color='lightblue')
            plt.title('Average Popularity by Genres')
            plt.ylabel('Average Popularity')
            plt.xlabel('Genre')
            plt.xticks(rotation=45)  # Rotate genre labels for better readability
            plt.tight_layout()  # Adjust layout to not cut off labels

            # Using Streamlit to display the plot
            st.pyplot(plt)  # Make sure to use plt after generating the plot

        # Example usage in Streamlit
        if st.button('Show Average Popularity by Genre'):
            # Assuming df_spo is already loaded and available in the Streamlit script
            # df_spo = pd.read_csv('your_dataset.csv')  # You can load your data similarly
            plot_average_popularity_by_genre(df_spo) 




        def calculate_MSE(models, data):
            results = {}
            X = data.select_dtypes(include=[np.number]).drop('popularity', axis=1)
            y = data['popularity']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

            for name, model in models.items():
                try:
                    with st.spinner(f'Training {name} model...'):
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        mse = mean_squared_error(y_test, y_pred)
                        results[name] = mse
                    st.success(f'{name} model trained successfully!')
                except Exception as e:
                    st.error(f"Failed to calculate MSE for {name}: {e}")

            return results

        # Set up models
        models = {
            'Lasso Regression': Pipeline([
                ('scaler', StandardScaler()),
                ('lasso', Lasso(alpha=0.01, random_state=42))
            ]),
            'Ridge Regression': make_pipeline(StandardScaler(), Ridge(alpha=2)),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }

        # Example usage in Streamlit
        if st.button('Calculate MSE for All Models'):
            # df_spo = pd.read_csv('your_dataset.csv')  # You can load your data similarly
            results = calculate_MSE(models, df_spo)
            if results:
                for model_name, mse in results.items():
                    st.write(f"Mean squared error of {model_name} is {mse}")

            explanation = """
            We can see MSE is too high for Ridge and Lasso, so I use another ML method: random forest.
            """
            st.write(explanation)

        

        def genre_classification(data):
            try:
                # Drop non-relevant columns and prepare features and labels
                X = data.drop(['genre', 'artist_name', 'track_name', 'track_id', 'key', 'mode', 'time_signature'], axis=1)
                y = data['genre']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

                # Inform the user that the model is training
                with st.spinner('Training genre classification model... This may take a few minutes.'):
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),  
                        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))  
                    ])

                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    result_form = classification_report(y_test, y_pred)

                st.success('Model trained successfully!')
                st.write("Accuracy:", accuracy)
                st.text(result_form)

            except Exception as e:
                st.error(f"Failed to classify genres: {e}")

        # Example usage in Streamlit
        if st.button('Classify Genres'):
            # Assuming df_spo is already loaded and available in the Streamlit script
            # df_spo = pd.read_csv('your_dataset.csv')  # You can load your data similarly
            genre_classification(df_spo) 




        manual_stopwords = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as",
    "until", "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
    "don", "should", "now", ":","/"
])

    def most_common_words(data):
        def remove_stopwords(text, stopwords_set):
            words = text.lower().split()
            filtered_words = [word for word in words if word not in stopwords_set]
            return " ".join(filtered_words)

        # 过滤和清理数据
        data = data.dropna(subset=['track_name'])
        track_names = data['track_name']
        data['track_name_cleaned'] = track_names.apply(remove_stopwords, args=(manual_stopwords,))
        
        # 计算词频
        word_freq = Counter(" ".join(data['track_name_cleaned']).split())
        most_common_words = word_freq.most_common(20)
        
        return most_common_words


    if st.button('Show Most Common Words in Track Names'):
        # 假设 df_spo 已经加载并可用
        # df_spo = pd.read_csv('your_dataset.csv')
        common_words = most_common_words(df_spo)
        st.write("The most common words in track names:")
        for word, freq in common_words:
            st.write(f"{word}: {freq}")


    def artist_analysis(data, artists):
        try:
            artist_counts = data['artist_name'].value_counts()
            st.write("Most Featured Artists:")
            st.write(artist_counts.head(10))
            
            features = data.groupby('artist_name')[['danceability', 'energy', 'tempo']].mean()
            sorted_danceability = features.sort_values(by='danceability', ascending=False)
            sorted_energy = features.sort_values(by='energy', ascending=False)
            sorted_tempo = features.sort_values(by='tempo', ascending=False)
            
            st.write("Artists with highest average danceability:")
            st.dataframe(sorted_danceability.head(10))
            st.write("Artists with highest average energy:")
            st.dataframe(sorted_energy.head(10))
            st.write("Artists with highest average tempo:")
            st.dataframe(sorted_tempo.head(10))
            
            for artist in artists:
                specific_artist = data[data['artist_name'] == artist]
                st.write(f"Pairplot for {artist}")
                pairplot_fig = sns.pairplot(specific_artist[['danceability', 'energy', 'tempo']])
                st.pyplot(pairplot_fig)
        
        except Exception as e:
            st.error(f"Error in artist analysis: {e}")

    # Assuming df_spo is already loaded
    if st.button('Analyze Artists'):
        artist_analysis(df_spo, ['Ariana Grande', 'Post Malone', 'Daddy Yankee'])




    def plot_duration_by_genre(data):
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_title("Duration of songs in different genres")
            sns.barplot(x='duration_ms', y='genre', color='lightblue', data=data, ax=ax)
            
            # 使用ax.bar_label来标注条形图
            for bars in ax.containers:
                ax.bar_label(bars)

            ax.set_xlabel('Duration (ms)')
            ax.set_ylabel('Genre')
            
            # 使用传递给st.pyplot的fig对象
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error plotting duration by genre: {e}")

    if st.button('Show Duration by Genre'):
        # 假设df_spo是已经加载的数据集
        plot_duration_by_genre(df_spo)
        



def page_top_songs():
    st.title("Top Songs by Artists")
    
    # Use your Streamlit secrets for these values or environment variables
    client_id = st.secrets["SPOTIFY_CLIENT_ID"]
    client_secret = st.secrets["SPOTIFY_CLIENT_SECRET"]
    
    token = get_token(client_id, client_secret)
    if token:
        artist_names = ["Ariana Grande", "Post Malone", "Daddy Yankee"]
        for artist_name in artist_names:
            artist_id = search_for_artist(token, artist_name)
            if artist_id:
                songs = get_songs_by_artist(token, artist_id)
                st.subheader(f"Top songs by {artist_name}:")
                # Make sure to display something even if songs is empty
                if not songs:
                    st.write(f"No songs found for {artist_name}")
                    continue
                
                # We will display song names and their popularity
                for i, song in enumerate(songs[:10], start=1):  # Show top 10 songs
                    st.write(f"{i}. {song['name']} - Popularity: {song['popularity']}")
    else:
        st.error("Failed to authenticate with Spotify API")

        

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

# Main App
def main():
    page = st.sidebar.selectbox("Choose a page", ["Music Analysis", "Top Songs", "Answer the Questions"])
    if page == "Music Analysis":
        page_music_analysis()
    elif page == "Answer the Questions":
        page_answer_questions()
    elif page == "Top Songs":
        page_top_songs()

if __name__ == "__main__":
    main()
