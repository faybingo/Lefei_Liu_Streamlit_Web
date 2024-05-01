import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import googleapiclient.discovery
import pandas as pd

#Ariana Grande's data!!!
#URL:https://www.youtube.com/watch?v=KNtJGQkC-WI
api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = 'AIzaSyCAUjiNjOwo84-GHSfvMckcEQm1O2mJm1I'

youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=DEVELOPER_KEY)

request = youtube.commentThreads().list(
    part="snippet",
    videoId="KNtJGQkC-WI",
    maxResults=100
)

comments = []

response = request.execute()

# Get the comments from the response.
for item in response['items']:
    comment = item['snippet']['topLevelComment']['snippet']
    public = item['snippet']['isPublic']
    comments.append([
        comment['authorDisplayName'],
        comment['publishedAt'],
        comment['likeCount'],
        comment['textOriginal'],
        public
    ])

while (1 == 1):
  try:
    nextPageToken = response['nextPageToken']
  except KeyError:
   break
  nextPageToken = response['nextPageToken']
  # Create a new request object with the next page token.
  nextRequest = youtube.commentThreads().list(part="snippet", videoId="KNtJGQkC-WI", maxResults=100, pageToken=nextPageToken)
  # Execute the next request.
  response = nextRequest.execute()
  # Get the comments from the next response.
  for item in response['items']:
    comment = item['snippet']['topLevelComment']['snippet']
    public = item['snippet']['isPublic']
    comments.append([
        comment['authorDisplayName'],
        comment['publishedAt'],
        comment['likeCount'],
        comment['textOriginal'],
        public
    ])

df = pd.DataFrame(comments, columns=['author', 'updated_at', 'like_count', 'text','public'])
df.info()

print(response['items'][0])
print(df.head(10))
print(df['text'])
df.to_csv('youtube_ag.csv', index=False)


df1 = pd.read_csv('/Users/liulefei/Desktop/webapp/youtube_ag.csv')
df1['text'] = df1['text'].str.replace('Evan Peter', '')
comments1= df1['text'].astype(str).tolist()
comments_text1 = ' '.join(comments1)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(comments_text1)
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()



#Post Malone's youtube review data!!!
#URL:https://www.youtube.com/watch?v=VzEqqSUWNjw

api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = 'AIzaSyCAUjiNjOwo84-GHSfvMckcEQm1O2mJm1I'

youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=DEVELOPER_KEY)

request = youtube.commentThreads().list(
    part="snippet",
    videoId="VzEqqSUWNjw",
    maxResults=100
)
comments = []
response = request.execute()
# Get the comments from the response.
for item in response['items']:
    comment = item['snippet']['topLevelComment']['snippet']
    public = item['snippet']['isPublic']
    comments.append([
        comment['authorDisplayName'],
        comment['publishedAt'],
        comment['likeCount'],
        comment['textOriginal'],
        public
    ])

while (1 == 1):
  try:
    nextPageToken = response['nextPageToken']
  except KeyError:
   break
  nextPageToken = response['nextPageToken']
  # Create a new request object with the next page token.
  nextRequest = youtube.commentThreads().list(part="snippet", videoId="VzEqqSUWNjw", maxResults=100, pageToken=nextPageToken)
  # Execute the next request.
  response = nextRequest.execute()
  # Get the comments from the next response.
  for item in response['items']:
    comment = item['snippet']['topLevelComment']['snippet']
    public = item['snippet']['isPublic']
    comments.append([
        comment['authorDisplayName'],
        comment['publishedAt'],
        comment['likeCount'],
        comment['textOriginal'],
        public
    ])

df = pd.DataFrame(comments, columns=['author', 'updated_at', 'like_count', 'text','public'])
df.info()

print(response['items'][0])
print(df.head(10))
print(df['text'])
df.to_csv('youtube_pm.csv', index=False)


df2 = pd.read_csv('/Users/liulefei/Desktop/webapp/youtube_pm.csv')
comments2 = df2['text'].astype(str).tolist()
comments_text2 = ' '.join(comments2)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(comments_text2)
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()





#Daddy Yankee's data!!!
#URL:https://www.youtube.com/watch?v=VvFUraacW1c

api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = 'AIzaSyCAUjiNjOwo84-GHSfvMckcEQm1O2mJm1I'

youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=DEVELOPER_KEY)

request = youtube.commentThreads().list(
    part="snippet",
    videoId="VvFUraacW1c",
    maxResults=100
)

comments = []

# Execute the request.
response = request.execute()

# Get the comments from the response.
for item in response['items']:
    comment = item['snippet']['topLevelComment']['snippet']
    public = item['snippet']['isPublic']
    comments.append([
        comment['authorDisplayName'],
        comment['publishedAt'],
        comment['likeCount'],
        comment['textOriginal'],
        public
    ])

while (1 == 1):
  try:
    nextPageToken = response['nextPageToken']
  except KeyError:
   break
  nextPageToken = response['nextPageToken']
  # Create a new request object with the next page token.
  nextRequest = youtube.commentThreads().list(part="snippet", videoId="VvFUraacW1c", maxResults=100, pageToken=nextPageToken)
  # Execute the next request.
  response = nextRequest.execute()
  # Get the comments from the next response.
  for item in response['items']:
    comment = item['snippet']['topLevelComment']['snippet']
    public = item['snippet']['isPublic']
    comments.append([
        comment['authorDisplayName'],
        comment['publishedAt'],
        comment['likeCount'],
        comment['textOriginal'],
        public
    ])

df = pd.DataFrame(comments, columns=['author', 'updated_at', 'like_count', 'text','public'])
df.info()

print(response['items'][0])
print(df.head(10))
print(df['text'])
df.to_csv('youtube_dy.csv', index=False)


df3 = pd.read_csv('/Users/liulefei/Desktop/webapp/youtube_dy.csv')
comments3= df3['text'].astype(str).tolist()
comments_text3 = ' '.join(comments3)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(comments_text3)
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

