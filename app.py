import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")

@st.cache_data
def load():
    df = pd.read_csv("movies_clean.csv").dropna().reset_index(drop=True)
    matrix = TfidfVectorizer(stop_words="english", max_features=5000).fit_transform(df["tags"])
    return df, matrix
df, matrix = load()

def recommend(title, n=5):
    idx = df.index[df["title"] == title][0]
    scores = cosine_similarity(matrix[idx], matrix).ravel()
    top = scores.argsort()[::-1][1:n + 1]
    return [(df["title"][i], scores[i]) for i in top]

st.title("🎬 Movie Recommender")
st.caption("Content-based: TF-IDF on plot, genres, keywords, cast and director + cosine similarity.")

choice = st.selectbox("Pick a movie you like", df["title"].sort_values())
count = st.slider("How many recommendations?", 3, 10, 5)

if st.button("Recommend"):
    for rank, (name, score) in enumerate(recommend(choice, count), 1):
        st.write(f"**{rank}. {name}**  —  similarity {score:.2f}")
