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


#import data
# df_spo= pd.read_csv('/Users/liulefei/Desktop/webapp/data/total_data.csv')
script_path = os.path.dirname(__file__)
filename = 'total_data'
file_path = os.path.join(script_path, f'{filename}.csv')
df_spo = pd.read_csv(file_path)
print(df_spo.head(10))
print(df_spo)
print(df_spo.shape)

#data cleaning
print(pd.isnull(df_spo).sum())
df_spo.dropna(inplace=True)
print(df_spo.info())
duplicate = df_spo.duplicated()
duplicate_count = duplicate.sum()
print("Duplicate sum is:", duplicate_count)
#print all the columns
print(df_spo.columns)
#print the types of each column
print(df_spo.dtypes)
#count artists' number of songs
print(df_spo.artist_name.value_counts())
#summary for all variables
print(df_spo.describe())


#correlation of each variable
def plot_correlation(data):
    corr_df = data.drop(['genre', 'artist_name', 'track_name','track_id','key', 'mode','time_signature'], axis=1).corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm')
    plt.title('Correlation between each variable')
    plt.show()

plot_correlation(df_spo)

#barplot of number of genres
def plot_genre_dist(data):
    number_genre = data.genre.value_counts()
    plt.figure(figsize=(10, 6))
    number_genre.plot(kind='bar', color='blue')
    plt.ylabel('Number of songs')
    plt.title('Distribution of genres')
    plt.xlabel('Genres')
    plt.show()

plot_genre_dist(df_spo)


#histgram of popularity
def plot_popularity_distribution(data):
    plt.figure(figsize=(10, 6))
    plt.hist(data['popularity'], bins=20, color='lightblue', edgecolor='black')
    plt.xlabel('Popularity')
    plt.ylabel('Numbers')
    plt.title('Distribution of popularity')
    plt.show()

plot_popularity_distribution(df_spo)
# #this is a skewed distribution.

#most popular songs 
def most_popular_songs(data):
    songs = df_spo[df_spo['popularity'] > 90]
    most_popular_songs = songs.sort_values('popularity', ascending=False).head(10)#to see first 10 songs
    return most_popular_songs

def plot_most_popular_songs(most_popular_songs):
    # 绘制条形图
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=most_popular_songs, x='popularity', palette='viridis')
    plt.title('Most popular songs')
    for bars in ax.containers:
        ax.bar_label(bars)
    plt.xlabel('Popularity')
    plt.ylabel('Number of songs')
    plt.show()

popular_songs = most_popular_songs(df_spo)
plot_most_popular_songs(popular_songs)

print(popular_songs)#we can see from this code, top 10 songs' artists are Ariana Grande,Post Malone and Daddy Yankee
#so I will extract their top 10 songs from spotify api, and compare their amazon album sales amount.



#Get top 10 songs of (Ariana Grande, Post Malone, Daddy Yankee) from api_scraper 
#code is the same with api_scraper.py
client_id='ec31934c1e8f48a788084e4210a2d355'
client_secret='ab60d2175d7347e79df96deffb125f6a'

#get token
def get_token():
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

#get token header
def get_auth_header(token):
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

#get three artists' songs:Ariana Grande, Post Malone, Daddy Yankee
artist_names = ["Ariana Grande", "Post Malone", "Daddy Yankee"]
token = get_token()

for artist_name in artist_names:
    result = search_for_artist(token, artist_name)

    if result:
        artist_id = result["id"]
        songs = get_songs_by_artist(token, artist_id)

        if songs:
            print(f"Top songs by {artist_name}:")
            for idx, song in enumerate(songs):
                print(f"{idx+1}. {song['name']}")
        else:
            print(f"Failed to retrieve songs for {artist_name}.")
    else:
        print(f"No artist named {artist_name} exists.")

    print()


script_path = os.path.dirname(__file__)
filename = 'artists_top10_data'
file_path = os.path.join(script_path, f'{filename}.csv')
df_tpo10 = pd.read_csv(file_path)

# #Then we want to see their sales amount of each singer on amazon page 1.
# #We first see Ariana Grande's sales amount
# # df_ag = pd.read_csv('/Users/liulefei/Desktop/webapp/amazon_ag.csv')
# script_path = os.path.dirname(__file__)
# filename_ag = 'amazon_ag'
# file_path_ag = os.path.join(script_path, f'{filename_ag}.csv')
# df_ag = pd.read_csv(file_path_ag)

# keys=df_ag.keys()
# col1_name = keys[0]
# col2_name= keys[1]
# df_ag = df_ag.rename(columns={col1_name: 'sales amount', col2_name: 'price'})
# df_ag['sales amount'] = df_ag['sales amount'].str.replace(r'1K+','1000+')
# df_ag['sales amount'] = df_ag['sales amount'].str.replace(r'\D+','',regex=True)
# #str.replace()替换string中的特定子串。r'\D+' 是一个regular expression，表示匹配多个non-numeric string。regex=True表示正则表达式模式匹配。
# df_ag['sales amount'] = df_ag['sales amount'].astype(int)
# print("Ariana Grande's sales amount:")
# print(df_ag)
# sales_ag=df_ag['sales amount'].sum()
# print("Ariana Grande's average sales amount:",sales_ag)


# #df_pm = pd.read_csv('/Users/liulefei/Desktop/webapp/amazon_pm.csv')
# script_path = os.path.dirname(__file__)
# filename_pm = 'amazon_pm'
# file_path_pm = os.path.join(script_path, f'{filename_ag}.csv')
# df_pm = pd.read_csv(file_path_pm)

# keys=df_pm.keys()
# col1_name = keys[0]
# col2_name= keys[1]
# df_pm = df_pm.rename(columns={col1_name: 'sales amount', col2_name: 'price'})
# df_pm['sales amount'] = df_pm['sales amount'].str.replace(r'\D+','',regex=True)
# df_pm['sales amount'] = df_pm['sales amount'].astype(int)
# print("Ariana Grande's sales amount:")
# print(df_pm)
# sales_pm=df_pm['sales amount'].sum()
# print("Ariana Grande's average sales amount:",sales_pm)
# print()
# df_compare= pd.DataFrame({'Source': ['amazon_ag', 'amazon_pm'], 'Total Sales': [sales_ag, sales_pm]})
# print(df_compare)
# #we can compare the sales amount of these three singers.
# #But it seems like Daddy Yankee is an old star, 
# #so the sales amount per month of his album is less than 50. 
# #His amazon page does not display data below 50.  So we only comapre the sales amount of Ariana Grande and Post Malone.




# #Also, I did a wordcloud plot for these three artists, the text sources are from youtube MV comments.

# #Wordcloud for Ariana Grande
# script_path = os.path.dirname(__file__)
# filename1 = 'youtube_ag'
# file_path1 = os.path.join(script_path, f'{filename1}.csv')
# df1 = pd.read_csv(file_path1)
# # df1 = pd.read_csv('/Users/liulefei/Desktop/webapp/youtube_ag.csv')#upload api data
# df1['text'] = df1['text'].str.replace('Evan Peter', '')
# comments1= df1['text'].astype(str).tolist()
# comments_text1 = ' '.join(comments1)
# wordcloud = WordCloud(width=800, height=400, background_color='white').generate(comments_text1)
# plt.figure(figsize=(10, 8))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.show()


# #wordcloud for Post Malone
# script_path = os.path.dirname(__file__)
# filename2 = 'youtube_pm'
# file_path2 = os.path.join(script_path, f'{filename2}.csv')
# df2 = pd.read_csv(file_path2)
# # df2 = pd.read_csv('/Users/liulefei/Desktop/webapp/youtube_pm.csv')
# comments2 = df2['text'].astype(str).tolist()
# comments_text2 = ' '.join(comments2)
# wordcloud = WordCloud(width=800, height=400, background_color='white').generate(comments_text2)
# plt.figure(figsize=(10, 8))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.show()


# #wordcloud for Daddy Yankee
# script_path = os.path.dirname(__file__)
# filename3 = 'youtube_dy'
# file_path3 = os.path.join(script_path, f'{filename3}.csv')
# df3 = pd.read_csv(file_path3)
# # df3 = pd.read_csv('/Users/liulefei/Desktop/webapp/youtube_dy.csv')
# comments3= df3['text'].astype(str).tolist()
# comments_text3 = ' '.join(comments3)
# wordcloud = WordCloud(width=800, height=400, background_color='white').generate(comments_text3)
# plt.figure(figsize=(10, 8))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.show()




# #barplot of popularity by genre
# def plot_average_popularity_by_genre(data):
#     # unique_genres_df = pd.DataFrame(data['genre'].unique(), columns=['Genre'])
#     gen = (data.groupby('genre')['popularity']
#            .mean()
#            .sort_values(ascending=False)
#            .reset_index()
#            .rename(columns={'popularity': 'Average popularity'}))
    
#     plt.figure(figsize=(10, 6))
#     plt.bar(gen['genre'], gen['Average popularity'], color='lightblue')
#     plt.title('Average popularity by genres')
#     plt.ylabel('Average popularity')
#     plt.xlabel('Genre')
#     plt.show()

# plot_average_popularity_by_genre(df_spo)




# #Lasso regression
# def calculate_MSE_lasso(data):
#     X = data.select_dtypes(include=[np.number]).drop('popularity', axis=1)
#     y = data['popularity']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
#     pipeline = Pipeline([
#         ('scaler', StandardScaler()),
#         ('lasso', Lasso(alpha=0.01, random_state=42)) 
#     ])
#     pipeline.fit(X_train, y_train)
#     y_pred = pipeline.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     print(f"Mean squared error of lasso model is {mse}")

# calculate_MSE_lasso(df_spo)



# #Ridge regression
# def calculate_MSE_ridge(data):
#     X = data.select_dtypes(include=[np.number]).drop(['popularity'], axis=1)  
#     y = data['popularity']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
#     pipeline = make_pipeline(StandardScaler(), Ridge(alpha=2))
#     pipeline.fit(X_train, y_train)
#     y_pred = pipeline.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     print(f"Mean squared error of ridge model is: {mse}")

# calculate_MSE_ridge(df_spo)
# #MSE is too high for Ridge and Lasso, so I use another ML method: random forest





# # Random forest model:It will take 5-7 minutes to run!!!!!!!!! MSE is approximately 152.
# # drop non-numeric columns and rows with missing values
# def cal_MSE_random_forest(data):
#     df_spo_drop = data.select_dtypes(include=['number']).dropna()
#     X = df_spo_drop.drop('popularity', axis=1)
#     y = df_spo_drop['popularity']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
#     rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
#     rf_regressor.fit(X_train, y_train)
#     y_pred = rf_regressor.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     print(f"Mean squared error for random forest model is {mse}")

# cal_MSE_random_forest(df_spo)





# #Genre classification: predict the genre of a track based on its audio features by creating a classification model.
# #it will take several minutes to run
# def genre_classification(data):
#     # Define features and target variable
#     X = data.drop(['genre', 'artist_name', 'track_name', 'track_id', 'key', 'mode', 'time_signature'], axis=1)
#     y = data['genre']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
#     pipeline = Pipeline([
#         ('scaler', StandardScaler()),  
#         ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))  
#     ])

#     pipeline.fit(X_train, y_train)
#     y_pred = pipeline.predict(X_test)
#     accuracy = 1 - accuracy_score(y_test, y_pred)
#     result_form = classification_report(y_test, y_pred)
#     print("Accuracy:", accuracy)
#     print(result_form)

# genre_classification(df_spo)






# # Most common words in track_names(use NLP) 
# # It should use NLTK package to find stopwords,but I cannot import it, I manually add some stopwords.
# # These are the common stopwords generated by chatgpt

# manual_stopwords = set([
#     "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
#     "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
#     "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
#     "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
#     "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
#     "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as",
#     "until", "while", "of", "at", "by", "for", "with", "about", "against", "between",
#     "into", "through", "during", "before", "after", "above", "below", "to", "from",
#     "up", "down", "in", "out", "on", "off", "over", "under", "again", "further",
#     "then", "once", "here", "there", "when", "where", "why", "how", "all", "any",
#     "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
#     "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
#     "don", "should", "now",
#     ":"  
# ])

# def most_common_words(data):
#     def remove_stopwords(text, stopwords_set):
#         if isinstance(text, str):  
#             words = text.lower().split()
#             filtered_words = [word for word in words if word not in stopwords_set]
#             return " ".join(filtered_words)
#         else:
#             return ""  

#     data = data.dropna(subset=['track_name'])
#     track_names = data['track_name']
#     data['track_name_cleaned'] = track_names.apply(remove_stopwords, args=(manual_stopwords,))
#     word_freq = Counter(data['track_name_cleaned'])
#     print("The most common words in track_names:")
#     for word, freq in word_freq.most_common(20):
#         print(f"{word}: {freq}")
# most_common_words(df_spo)




# # Count the occurrences of those three artists and sort 
# def artist_analysis(data, artists):
#     artist_counts = data['artist_name'].value_counts()
#     print("Most Featured Artists:")
#     print(artist_counts.head(10))
    
#     features = data.groupby('artist_name')[['danceability', 'energy', 'tempo']].mean()
#     sorted_danceability = features.sort_values(by='danceability',ascending=False)
#     sorted_energy = features.sort_values(by='energy',ascending=False)
#     sorted_tempo= features.sort_values(by='tempo',ascending=False)

#     print("Artists with highest average danceability:")
#     print(sorted_danceability.head(10))
#     print("Artists with highest average energy:")
#     print(sorted_energy.head(10))
#     print("Artists with highest average tempo:")
#     print(sorted_tempo.head(10))
    
#     for artist in artists:
#         specific_artist = data[data['artist_name'] == artist]  
#         sns.pairplot(specific_artist[['danceability','energy','tempo']])
#         plt.show()

# artist_analysis(df_spo, ['Ariana Grande', 'Post Malone', 'Daddy Yankee'])



# def plot_duration_by_genre(data):
#     plt.figure(figsize=(8, 6))
#     plt.title("Duration of songs in different genres")
#     ax = sns.barplot(x='duration_ms', y='genre', color='lightblue', data=data)
#     for bars in ax.containers:
#         ax.bar_label(bars)
#     plt.xlabel('Duration (ms)')
#     plt.ylabel('Genre')
#     plt.show()

# plot_duration_by_genre(df_spo)






