import ast
import pandas as pd

movies = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv").drop(columns=["title"])
df = movies.merge(credits, left_on="id", right_on="movie_id")

def names(text, limit=None):
    items = [i["name"] for i in ast.literal_eval(text)]
    return items[:limit] if limit else items

def director(text):
    for p in ast.literal_eval(text):
        if p["job"] == "Director":
            return [p["name"]]
    return []

def squash(words):
    return [w.replace(" ", "") for w in words]

df["overview"] = df["overview"].fillna("")
df["tags"] = (
    df["overview"] + " "
    + df["genres"].apply(lambda x: " ".join(squash(names(x)))) + " "
    + df["keywords"].apply(lambda x: " ".join(squash(names(x)))) + " "
    + df["cast"].apply(lambda x: " ".join(squash(names(x, 3)))) + " "
    + df["crew"].apply(lambda x: " ".join(squash(director(x))))
).str.lower()

df[["title", "tags"]].drop_duplicates("title").to_csv("data/movies_clean.csv", index=False)
print("Saved", len(df), "movies to data/movies_clean.csv")
