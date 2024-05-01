import requests
import json
import base64

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

