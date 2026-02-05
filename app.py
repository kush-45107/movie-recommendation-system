from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

vectorizer = joblib.load('tfidf_vectorizer.pkl')
similarity = joblib.load('similarity.pkl')
data = joblib.load('movies_data.pkl')

def recommend(movie_name):
    movie_name = movie_name.strip().lower()
    idx = data[data['Title'].str.lower() == movie_name].index
    if len(idx) == 0:
        return []

    idx = idx[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]

    recommendations = []
    for i, _ in scores:
        movie_info = {
            'title': data.iloc[i]['Title'],
            'actor1': data.iloc[i]['Actors 1'],
            'actor2': data.iloc[i]['Actors 2'],
            'actor3': data.iloc[i]['Actors 3'],
            'director': data.iloc[i]['Director'],
            'producer': data.iloc[i]['Producer']
        }
        recommendations.append(movie_info)

    return recommendations

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    not_found = False
    if request.method == "POST":
        movie = request.form["movie"]
        recommendations = recommend(movie)
        if not recommendations:
            not_found = True
    return render_template("index.html", recommendations=recommendations, not_found=not_found)


if __name__ == "__main__":
    app.run(debug=True)
