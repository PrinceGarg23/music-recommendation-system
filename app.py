import requests

API_KEY = "fd3b4e20822285455da5b983b1764624"
USERNAME = "PrinceGarg23"

def get_top_tracks(limit=50):
    url = f"http://ws.audioscrobbler.com/2.0/?method=user.gettoptracks&user={USERNAME}&api_key={API_KEY}&limit={limit}&format=json"
    response = requests.get(url)
    data = response.json()
    #print(data)
    return data["toptracks"]["track"]

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_data(tracks):
    track_names = [track.get("name", "") for track in top_tracks]
    if not track_names:
        print("Track Names is empty. Please check the data.")
        return None
    # Custom tokenizer to handle None or empty inputs and remove empty strings
    def custom_tokenizer(text):
        return [word for word in text.split() if word] if text else []
    
    vectorizer = TfidfVectorizer(stop_words='english', tokenizer=custom_tokenizer).fit(track_names)
    similarity_matrix = cosine_similarity(vectorizer.transform(track_names))
    return similarity_matrix


def recommend_songs(similarity_matrix, song_name, n_songs=5):
    try:
        song_idx = track_names.index(song_name)
        print(song_idx)
    except ValueError:
        return []

    similarity_scores = list(enumerate(similarity_matrix[song_idx]))
    print(similarity_matrix)
    print(similarity_scores)
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    print(similarity_scores)
    top_n_songs = similarity_scores[1:n_songs + 1]
    print(top_n_songs)
    similar_songs = []
    for idx, _ in top_n_songs:
        similar_songs.append(track_names[idx])
        print(track_names[idx])
    print(similar_songs)
    return similar_songs


import numpy as np

def find_similar_songs(song_idx, similarity_matrix, k=5):
    similar_song_indices = np.argsort(similarity_matrix[song_idx])[::-1][1:k+1]
    return similar_song_indices

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        song_name = request.form['song']
        similarity_matrix = preprocess_data(top_tracks)
        recommended_songs = recommend_songs(similarity_matrix, song_name)

        return render_template('recommendations.html', song_name=song_name, recommended_songs=recommended_songs)

    return render_template('index.html', songs=top_tracks)


if __name__ == "__main__":
    top_tracks = get_top_tracks()
    print(top_tracks)
    track_names = [track["name"] for track in top_tracks]
    if not top_tracks:
        print("Top Tracks is empty. Please make sure it contains data.")
    else:
        similarity_matrix = preprocess_data(top_tracks)
    app.run(debug=True)
