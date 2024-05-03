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
import googleapiclient.discovery
from sklearn.svm import SVR
import streamlit as st
import io
from wordcloud import WordCloud, STOPWORDS
import re

# Load data
def load_data():
    try:
        df_spo = pd.read_csv('total_data.csv')
        return df_spo
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

def load_artist_data():
    script_path = os.path.dirname(__file__)
    filename = 'artists_top10_data'
    file_path =os.path.join(script_path, f'{filename}.csv')
    df_top10= pd.read_csv(file_path)
    return df_top10

# Music Analysis Page
def page_music_analysis():
    df_spo = load_data()
    if df_spo is not None:
        st.title("Music Data Analysis")
        st.markdown("""
            Name : Lefei Liu

            ### How to Use This Web
            My web consists of 3 pages. The first page is 'Filter Data'. You can enter the name of an artist (for example, Ariana Grande), select your desired genres, and then adjust the sliders of other variables on the bottom left to obtain the data you want and click 'Apply Filters', you will see your desired data. The second page is 'Music Analysis', where interaction with the application is done by clicking buttons like "Show Data Cleaning Information" or "Show Correlation of Each Variable". Each button provides detailed insights or visualizations based on total data. You can click on each button to view its corresponding explanation of the meaning. The third page is for answering the 5 questions.
            
            ### Major Gotchas
            "Calculating MSE for Three Models" will take 4-5 minutes to complete. "Classifying Genres" may take 3-4 minutes to complete. If you encounter an error, just simply refresh the URL and it should resolve the issue.
        """)
        
        #data cleaning 
        if st.button('Show Data Cleaning Information'):
            with st.expander("Data Cleaning Information"):
                st.write("Missing values in each column:")
                st.write(df_spo.isnull().sum())
                duplicate_count = df_spo.duplicated().sum()
                st.write(f"Number of duplicate rows: {duplicate_count}")
                st.write("From the chart above, there is only one missing value in the 'track_name' column, which seems inconsequential.")

        #data description
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
                st.write("Here's a basic overview of the DataFrame 'df_spo'. The second form presents a statistical summary of the entire dataset. The third table show us the number of songs per artist with this dataset")

        # correlation
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
                st.write("We can see the correlation heatmap provides a basic understanding of the relationships between each variable.")

        #histogram
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
                st.write("We can observe the distribution of one column 'popularity' in the df_spo total dataset, which is a right-skewed histogram.")
                

        #most popular songs
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
                    st.pyplot(fig)  
                    st.write("The initial plot displays the top 10 most popular songs. It's worth mentioning that two of these songs have a popularity score of 100. ")
                popular_songs_df = most_popular_songs(df_spo)
                plot_most_popular_songs(popular_songs_df)
                st.write(popular_songs_df)
                st.write("In the second form, we can see that the names of these 10 songs and their respective artists. Obviously, the artists behind the top 10 songs include Ariana Grande, Post Malone, and Daddy Yankee.")
                st.write("I will extract their top 10 songs from the Spotify API and conduct further analysis based on these three artists.")
        
        

        #top 10 songs for three artists
        if st.button('Show Top 10 Songs'):
            with st.expander("Top 10 Songs Data"):
                try:
                    script_path = os.path.dirname(__file__)  
                    filename = 'artists_top10_data.csv'    
                    file_path = os.path.join(script_path, filename)
                    df_top10 = pd.read_csv(file_path, encoding='ISO-8859-1')     
                    st.dataframe(df_top10)                
                except Exception as e:
                    st.error(f"Failed to load data: {e}")  
                    st.write("This is the top10 songs data that I scraped from Spofity API")


        #compare sales amounts
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

            st.write("We scraped sales data from Amazon（API code is called 'amazon_scraper', and the corresponding datasets are named 'amazon_ag' and 'amazon_pm'.). The results clearly show that Ariana Grande's sales surpass those of Post Malone. Also, sales data for Daddy Yankee with monthly sales below 50 were omitted from Amazon. This observation suggests that Daddy Yankee might not currently enjoy the same level of popularity. Perhaps because our data is a bit outdated, the current sales amount do not match the previous music data. ")



        #wordclouds
        def generate_and_display_wordcloud(file_path, artist_name):
            try:
                df = pd.read_csv(file_path)
                comments = df['text'].astype(str).tolist()
                comments_text = ' '.join(comments)

                stopwords = set(STOPWORDS)
                additional_stopwords = {'posty','some', 'word', 'here', 'song', 'make', 'think', 'songs', 'know', 'feel', 'music', 'video', 'always', 'will', 'Evan Peter', 'much', 'need', 'say'}
                stopwords.update(additional_stopwords)
                wordcloud = WordCloud(width=800, height=400, background_color='white',stopwords=stopwords).generate(comments_text)

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
                
            st.write("By scraping review data from YouTube for each artist and visualizing their wordclouds, we gain insights into audience feedback and interests. Analyzing the keywords in the word clouds helps us better understand what resonates with the audience, enabling us to optimize video content, enhance user experience, and devise more effective marketing strategies.")

            
        #avg pupularity for genre
        def plot_average_popularity_by_genre(data):
            # Grouping data by genre and calculating mean popularity
            gen = (data.groupby('genre')['popularity']
                .mean()
                .sort_values(ascending=False)
                .reset_index()
                .rename(columns={'popularity': 'Average popularity'}))
            
            plt.figure(figsize=(10, 6))
            plt.bar(gen['genre'], gen['Average popularity'], color='lightblue')
            plt.title('Average Popularity by Genres')
            plt.ylabel('Average Popularity')
            plt.xlabel('Genre')
            plt.tight_layout()  # Adjust layout to not cut off labels
            st.pyplot(plt)  # 

        if st.button('Show Average Popularity by Genre'):
            # df_spo = pd.read_csv('your_dataset.csv')  
            plot_average_popularity_by_genre(df_spo) 
            st.write("Based on different genres, we can obtain the popularity values for each genre. The most polular genre are 'Pop' and 'Rap'")


        #machine learning model and cal MSE
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

            st.write("Upon observing excessively high MSE values for Ridge and Lasso regression, I chose for an alternative ML method: random forest. We can see the MSE value significantly decreased.")


        
        #genrer classify
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
            st.write("This model uses the random forest algorithm to train a music type classification model, and can ultimately display the model's accuracy and classification report. 0.38 means that the model correctly predicted about 38% of the samples. It can be seen from the classification report that the prediction performance of most music types is low, and the precision, recall, and f1-score of many types are low, indicating that the model has large errors. Further optimization of the model may be required to improve classification accuracy.")



        #most common words
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
    "don", "should", "now", ":", "/","2","1","dont","ii","im","op","3","one","man","little"
])

        def most_common_words(data):
            def remove_stopwords(text):
                # Remove punctuation
                text = re.sub(r'[^\w\s]', '', text)
                words = text.lower().split()
                filtered_words = [word for word in words if word not in manual_stopwords]
                return " ".join(filtered_words)

            data = data.dropna(subset=['track_name'])
            track_names = data['track_name']
            data['track_name_cleaned'] = track_names.apply(remove_stopwords)

            word_freq = Counter(" ".join(data['track_name_cleaned']).split())
            most_common_words = word_freq.most_common(20)
            
            return most_common_words

    if st.button('Show Most Common Words in Track Names'):
        # df_spo = pd.read_csv('your_dataset.csv')
        common_words = most_common_words(df_spo)
        st.write("The most common words in track names:")
        for word, freq in common_words:
            st.write(f"{word}: {freq}")
        st.write("We want to count the most common words in music track names. First, manually generate some common stop words. Then, count the word frequency after removing stop words. Finally, it returns the 20 most common words in the track names and their frequencies.")
        st.write("We can analyze something through the results. 'Remix' and 'remastered' indicate that remixed or remastered versions of old songs are popular in the market. 'Live' indicates that the live version of the song is popular. The high frequency of the words 'love' and 'life' highlights the universality of emotions and life themes in the songs, which resonate with a wide range of listeners. The common Spanish words 'la' and 'de' indicate that a significant portion of the music may be aimed at the Spanish-speaking market.")

    #artist analysis
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

    if st.button('Analyze Artists'):
        artist_analysis(df_spo, ['Ariana Grande', 'Post Malone', 'Daddy Yankee'])
        st.write("In this module, we analyze artists. The first table shows the ten most frequent artists in the data. Tables 2-4 show the average artist's danceability, energy, and tempo, and show the top ten artists with the highest rankings in these metrics. Then, take our previous visual analysis of three popular singers (Ariana Grande, Post Malone, and Daddy Yankee) to compare their differences by plotting pairplots between danceability, energy, and tempo.")



    #genre
    def plot_duration_by_genre(data):
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_title("Duration of songs in different genres")
            sns.barplot(x='duration_ms', y='genre', color='lightblue', data=data, ax=ax)
            
            for bars in ax.containers:
                ax.bar_label(bars)
            ax.set_xlabel('Duration (ms)')
            ax.set_ylabel('Genre')
            
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error plotting duration by genre: {e}")

    if st.button('Show Duration by Genre'):
        plot_duration_by_genre(df_spo)
        st.write("This is a graph plotting song duration for different genres of music, with the x-axis representing song duration in milliseconds and the y-axis representing genre. The genres 'world' and 'classical' have the longest durations.")



    #reference
    if st.button('Reference'):
        st.markdown("""
        ## References
        - [Repository on GitHub1](https://github.com/darshilparmar/amazon-web-scraping-python-project) 
        - [Repository on GitHub2](https://github.com/analyticswithadam/Python/blob/main/YouTube_Comments_Advanced.ipynb) 
        - [Repository on GitHub3](https://github.com/abhashpanwar/autoscraper)
        - [Repository on GitHub4](https://github.com/EuanMorgan/SpotifyDiscoverWeeklyRescuer)
                    
    """)





def page_filter_data():
    st.title("Filtered Data")
    df_spo = load_data()

    st.sidebar.header("Filter Options")
    artist_name = st.sidebar.text_input("Artist Name")
    genre = st.sidebar.multiselect("Genre",options=df_spo['genre'].unique(), default=df_spo['genre'].unique())
    tempo_range = st.sidebar.slider("Tempo",int(df_spo['tempo'].min()), int(df_spo['tempo'].max()), (int(df_spo['tempo'].min()), int(df_spo['tempo'].max())))
    valence_range = st.sidebar.slider("Valence", float(df_spo['valence'].min()),float(df_spo['valence'].max()), (float(df_spo['valence'].min()), float(df_spo['valence'].max())))
    acousticness_range = st.sidebar.slider("Acousticness",float(df_spo['acousticness'].min()), float(df_spo['acousticness'].max()), (float(df_spo['acousticness'].min()), float(df_spo['acousticness'].max())))
    danceability_range = st.sidebar.slider("Danceability",float(df_spo['danceability'].min()), float(df_spo['danceability'].max()), (float(df_spo['danceability'].min()), float(df_spo['danceability'].max())))
    energy_range = st.sidebar.slider("Energy",float(df_spo['energy'].min()), float(df_spo['energy'].max()), (float(df_spo['energy'].min()), float(df_spo['energy'].max())))
    liveness_range = st.sidebar.slider("Liveness",float(df_spo['liveness'].min()),float(df_spo['liveness'].max()), (float(df_spo['liveness'].min()), float(df_spo['liveness'].max())))
    loudness_range = st.sidebar.slider("Loudness",float(df_spo['loudness'].min()),float(df_spo['loudness'].max()), (float(df_spo['loudness'].min()), float(df_spo['loudness'].max())))
    speechiness_range = st.sidebar.slider("Speechiness",float(df_spo['speechiness'].min()),float(df_spo['speechiness'].max()), (float(df_spo['speechiness'].min()), float(df_spo['speechiness'].max())))


    if st.sidebar.button("Apply Filters"):
        filtered_df = df_spo
        if artist_name:
            filtered_df = filtered_df[filtered_df['artist_name'].str.contains(artist_name, case=False, na=False)]
        if genre:
            filtered_df = filtered_df[filtered_df['genre'].isin(genre)]
        filtered_df = filtered_df[(filtered_df['tempo'] >= tempo_range[0]) & (filtered_df['tempo'] <= tempo_range[1])]
        filtered_df = filtered_df[(filtered_df['valence'] >= valence_range[0])&(filtered_df['valence'] <= valence_range[1])]
        filtered_df = filtered_df[(filtered_df['acousticness'] >= acousticness_range[0])& (filtered_df['acousticness'] <= acousticness_range[1])]
        filtered_df = filtered_df[(filtered_df['danceability'] >= danceability_range[0])& (filtered_df['danceability'] <= danceability_range[1])]
        filtered_df = filtered_df[(filtered_df['energy'] >= energy_range[0]) &(filtered_df['energy'] <= energy_range[1])]
        filtered_df = filtered_df[(filtered_df['liveness'] >= liveness_range[0]) &(filtered_df['liveness'] <= liveness_range[1])]
        filtered_df = filtered_df[(filtered_df['loudness'] >= loudness_range[0])& (filtered_df['loudness'] <= loudness_range[1])]
        filtered_df = filtered_df[(filtered_df['speechiness'] >= speechiness_range[0])&(filtered_df['speechiness'] <= speechiness_range[1])]

        st.dataframe(filtered_df)
    else:
        # Display the full dataframe by default
        st.dataframe(df_spo)
        



def page_answer_questions():
    st.title("Answer the Last 5 Questions")
    st.markdown("""
        **4. What did you set out to study?**  
        - **Data visualization:** Plot a heatmap of the correlation matrix to observe the relationships between numerical variables; generate a histogram of music popularity to illustrate the distribution of popularity; Creat a bar plot showing the distribution of music genres to visualize the quantity distribution of different genres; plot a bar chart of average popularity by genre to showcase the average popularity of each genre; show the number of songs in different genres and their popularity distributions in a barchart; display a list of the most popular songs and further analyzes their popularity.
        - **Regression analysis:** Use Lasso and Ridge regression models to evaluate the relationship between each variable and popularity; Use random forest to compare and to find a better model by using MSE.
        - **Classification and prediction of music genres:** Use random forest algorithm to train a music type classification model, and get model's accuracy and classification report.
        - **Song characteristics Analysis:** Analyze the performance of songs by specific artists (such as Ariana Grande, Post Malone, Daddy Yankee) with respective to their danceability, energy, and tempo. Perform pairplot visualization on these artists' songs.
        - **Artists analysis:** Identify the most popular songs and displayed the top 10 songs based on popularity. Retrieved the artists of these songs by using Spotify API. Compare their amazon sales amounts to see the performance of the singer in the market. And scrape comments from their YouTube MVs to create wordclouds. Analyze the occurrence frequency of artists and displayed the top 10 artists with the highest occurrence frequency.
        - **Text Analysis:** Conduct wordcloud analysis of YouTube music video comments to identify the most common words. It can help us to understand audience feedback and emotional tendencies.

        **5. What did you discover/what were your conclusions?**  
        For each analysis, I've provided the conclusions directly below the corresponding code results. Please go back to see page 'Music Analysis'.

        **6. What difficulties did you have in completing the project?**  
        After writing the Streamlit code, it could not be deployed successfully, which was a big problem. I searched a lot of information for a long time and found that I need to add a requirement.txt.

        **7. What skills did you wish you had while you were doing the project?**  
        I learned how to obtain an API, learned to scrape any web page, and finally made such a web app. This is something I've never done before. I am so excited and happy to learn that!

        **8. What would you do "next" to expand or augment the project?**  
        I would use NLP techniques to analyze review text data, extracting valuable insights into user preferences. By investigating the data, I want to explore more machine learning algorithms (such as support vector machine) to uncover hidden patterns and information. This comprehensive approach will enable me to gain a deeper understanding of user behavior and preferences, facilitating more effective decision-making.
    """)


# Main App
def main():
    page = st.sidebar.selectbox("Choose a page", ["Filter Data", "Music Analysis", "Answer the Last 5 Questions"])
    
    if page == "Filter Data":
        page_filter_data()
    elif page == "Music Analysis":
        page_music_analysis()
    elif page == "Answer the Last 5 Questions":
        page_answer_questions()

if __name__ == "__main__":
    main()
