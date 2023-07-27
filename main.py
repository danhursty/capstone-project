import json
import uuid
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import openai
from flask import Flask, render_template, request
from dotenv import load_dotenv
import os


# Create Flask app
app = Flask(__name__)

# Load API key
def load_api_key() -> str:
    load_dotenv()
    return os.getenv('OPENAI_API_KEY')

# Load dataframes
def load_dataframes() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_new = pd.read_pickle('movies.pkl')
    df_old = pd.read_csv('imdb_top_1000.csv')
    df_old = df_old.rename(columns={'Series_Title': 'Movie_Title'})
    df_old['Gross'] = df_old['Gross'].fillna(0)
    df_old = df_old.dropna()
    return df_new, df_old

# Process dataframes
def process_dataframe(df_new: pd.DataFrame, df_old: pd.DataFrame) -> pd.DataFrame:
    df_new_poster = pd.merge(df_new, df_old[['Movie_Title', 'Poster_Link']], on='Movie_Title', how='left')
    df_new_poster = df_new_poster.explode('Genre')
    df_new_poster['Genre'] = df_new_poster['Genre'].str.replace(' ', '')
    return df_new_poster

# Get genre embeddings
def get_genre_embeddings(df: pd.DataFrame, model: str='text-embedding-ada-002') -> List[Dict[str, Any]]:
    df_genre = df['Genre'].unique().tolist()
    embeddings = []
    for genre in df_genre:
        response = openai.Embedding.create(input=genre, model=model)
        embeddings.append({
            "genre": genre,
            "embedding": response['data'][0]['embedding']
        })
    return embeddings

# Calculate cosine similarities
def vector_similarity(x: List[float], y: List[float]) -> float:
    return np.dot(np.array(x), np.array(y))

# Calculate cosine similarities
def calculate_cosine_similarities(user_embedding: List[float], embeddings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cosine_similarities = []
    for embedding in embeddings:
        cosine_similarity = vector_similarity(user_embedding, embedding['embedding'])
        cosine_similarities.append({
            "genre": embedding['genre'],
            "cosine_similarity": cosine_similarity
        })
    return cosine_similarities

# Get recommendations
def get_recommendations(df: pd.DataFrame, most_similar_genre: Dict[str, Any]) -> pd.DataFrame:
    df_recommendation = df.loc[df['Genre'] == most_similar_genre['genre'], ['Movie_Title', 'Genre', 'IMDB_Rating', 'Poster_Link']].sample(5)
    df_recommendation = df_recommendation.drop_duplicates()
    df_recommendation = df_recommendation.sort_values(by='IMDB_Rating', ascending=False)
    return df_recommendation

# Save data
def save_data(user_query: str, most_similar_genre: Dict[str, Any], df_recommendation: pd.DataFrame) -> None:
    try:
        with open('data.json', 'r') as infile:
            existing_data = json.load(infile)
    except FileNotFoundError:
        existing_data = []

    new_data = {
        'uuid': str(uuid.uuid4()),
        'user_query': user_query,
        'Genre': most_similar_genre['genre'],
        'df_recommendation': df_recommendation.to_dict(orient='records')
    }
    all_data = existing_data + [new_data]
    with open('data.json', 'w') as outfile:
        json.dump(all_data, outfile)

# Create route
@app.route('/', methods=['GET', 'POST'])
def recommendation():
    if request.method == 'POST':
        user_query = request.form['user_query']
        user_embedding = openai.Embedding.create(input=user_query, model="text-embedding-ada-002")['data'][0]['embedding']
        openai.api_key = load_api_key()
        df_new, df_old = load_dataframes()
        df_new_poster = process_dataframe(df_new, df_old)
        embeddings = get_genre_embeddings(df_new_poster)
        # Calculate cosine similarities
        cosine_similarities = calculate_cosine_similarities(user_embedding, embeddings)
        df_cosine_similarities = pd.DataFrame(cosine_similarities)
        df_cosine_similarities = df_cosine_similarities.sort_values(by='cosine_similarity', ascending=False).head(5)
        most_similar_genre = max(cosine_similarities, key=lambda x: x['cosine_similarity'])

        # Get recommendations
        df_recommendation = get_recommendations(df_new_poster, most_similar_genre)
        # save_data(user_query, most_similar_genre, df_recommendation)
        return render_template('index.html', cosine_similarities=df_cosine_similarities, user_query=user_query, df_recommendation=df_recommendation)

    return render_template('index.html')

# Run app
if __name__ == '__main__':
    # openai.api_key = load_api_key()
    # df_new, df_old = load_dataframes()
    # df_new_poster = process_dataframe(df_new, df_old)
    # embeddings = get_genre_embeddings(df_new_poster)
    app.run(port=5000)
