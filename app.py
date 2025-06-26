import pandas as pd
import pickle
import requests
from flask import Flask, render_template, request, jsonify
from datetime import datetime
from urllib.parse import quote
import gzip
from scipy import sparse
import numpy as np
import time

app = Flask(__name__)

# TMDB Configuration
TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/"
TMDB_API_BASE = "https://api.themoviedb.org/3"

# Cache for trending movies
trending_movies_cache = {
    'movies': [],
    'last_updated': 0
}

# Load the preprocessed data
try:
    movies = pd.read_pickle('models/processed_movies.pkl')
    with gzip.open('models/cosine_sim.pkl.gz', 'rb') as f:
        cosine_sim = pickle.load(f)
        if sparse.issparse(cosine_sim):
            cosine_sim = cosine_sim.toarray()
        cosine_sim = cosine_sim.astype(np.float32)
    with open('models/indices.pkl', 'rb') as f:
        indices = pickle.load(f)
    print("Data loaded successfully!")
except Exception as e:
    print(f"Error loading data: {e}")
    raise


def get_trending_movies():
    """Get trending movies from TMDB API with caching"""
    try:
        # Check if cache is still valid (1 hour)
        current_time = time.time()
        if current_time - trending_movies_cache['last_updated'] < 3600 and trending_movies_cache['movies']:
            return trending_movies_cache['movies']

        url = f"{TMDB_API_BASE}/trending/movie/day?api_key={TMDB_API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        trending_movies = response.json()['results'][:10]  # Get top 10 trending movies

        # Process the trending movies to match our format
        processed_movies = []
        for movie in trending_movies:
            # Skip movies without posters for better UI
            if not movie.get('poster_path'):
                continue

            processed_movies.append({
                'title': movie.get('title', 'Unknown'),
                'poster': f"{TMDB_IMAGE_BASE}w500{movie['poster_path']}",
                'release_date': movie.get('release_date', 'Unknown'),
                'tmdb_id': movie.get('id'),
                'vote_average': movie.get('vote_average', 0)
            })

        # Update cache
        trending_movies_cache['movies'] = processed_movies[:10]  # Return max 10 movies
        trending_movies_cache['last_updated'] = int(current_time)

        return trending_movies_cache['movies']

    except requests.exceptions.RequestException as e:
        print(f"TMDB API request failed: {e}")
        # Return cached movies even if stale when API fails
        return trending_movies_cache['movies']
    except Exception as e:
        print(f"Error processing trending movies: {e}")
        return []


def get_movie_poster(movie_id, title):
    """Get movie poster with multiple fallback methods"""
    try:
        # Method 1: Try TMDB API with movie ID
        url = f"{TMDB_API_BASE}/movie/{movie_id}?api_key={TMDB_API_KEY}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get('poster_path'):
            return f"{TMDB_IMAGE_BASE}w500{data['poster_path']}"
    except:
        pass

    try:
        # Method 2: Search TMDB by title
        url = f"{TMDB_API_BASE}/search/movie?api_key={TMDB_API_KEY}&query={quote(title)}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data['results'] and data['results'][0].get('poster_path'):
            return f"{TMDB_IMAGE_BASE}w500{data['results'][0]['poster_path']}"
    except:
        pass

    # Method 3: Check local dataset
    movie_data = movies[movies['id'] == movie_id]
    if not movie_data.empty and pd.notna(movie_data.iloc[0]['poster_path']):
        return f"{TMDB_IMAGE_BASE}w500{movie_data.iloc[0]['poster_path']}"

    return None  # No poster found



def get_cast_photos(movie_id, cast_list):
    """Get cast photos with multiple fallback methods"""
    try:
        # Try to get fresh cast data from TMDB API
        url = f"{TMDB_API_BASE}/movie/{movie_id}/credits?api_key={TMDB_API_KEY}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        credits = response.json()

        updated_cast = []
        for actor in cast_list[:5]:  # Only top 5 cast members
            # Find matching actor in API response
            api_actor = next((a for a in credits['cast'] if a['name'] == actor['name']), None)
            profile_path = api_actor['profile_path'] if api_actor else None

            updated_cast.append({
                'name': actor['name'],
                'character': actor['character'],
                'photo': f"{TMDB_IMAGE_BASE}w185{profile_path}" if profile_path else None,
                'wikipedia_summary': get_wikipedia_summary(actor['name'])
            })
        return updated_cast
    except:
        # Fallback to original cast data if API fails
        return [{
            'name': actor['name'],
            'character': actor['character'],
            'photo': None,
            'wikipedia_summary': get_wikipedia_summary(actor['name'])
        } for actor in cast_list[:5]]


def format_runtime(minutes):
    """Convert minutes to hours and minutes format"""
    try:
        minutes = int(minutes)
        return f"{minutes // 60}h {minutes % 60}m"
    except:
        return f"{minutes} min"


def get_recommendations(title):
    """Get movie recommendations based on cosine similarity"""
    try:
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        return movies.iloc[movie_indices]
    except Exception as e:
        print(f"Recommendation error: {e}")
        return None


def get_wikipedia_summary(name):
    """Get a brief Wikipedia summary for a person"""
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(name.replace(' ', '_'))}"
        response = requests.get(url, timeout=3)
        response.raise_for_status()
        data = response.json()
        return data.get('extract', 'No information available')
    except:
        return "No information available"



@app.route('/')
def home():
    suggestions = movies['title_y'].tolist()
    trending_movies = get_trending_movies()

    # Fallback: If no trending movies, use some popular movies from our dataset
    if not trending_movies:
        trending_movies = movies.sort_values('popularity', ascending=False).head(10)
        trending_movies = [{
            'title': row['title_y'],
            'poster': get_movie_poster(row['id'], row['title_y']),
            'release_date': row['release_date'],
            'tmdb_id': row['id'],
            'vote_average': row['vote_average']
        } for _, row in trending_movies.iterrows()]

    return render_template('index.html',
                           suggestions=suggestions,
                           trending_movies=trending_movies,
                           show_trending=True)

@app.route('/get-suggestions')
def get_suggestions():
    query = request.args.get('query', '').lower()
    if not query:
        return jsonify([])

    # Filter movies that contain the query
    suggestions = movies[movies['title_y'].str.lower().str.contains(query)]['title_y'].tolist()
    return jsonify(suggestions[:10])  # Return top 10 matches


@app.route('/recommend', methods=['GET'])
def recommend():
    title = request.args.get('movie')
    if not title:
        return render_template('index.html', error="Please enter a movie name")

    try:
        # Get the base movie data from our dataset
        movie_data = movies[movies['title_y'] == title].iloc[0]
        movie_id = movie_data['id']

        # Get poster with multiple fallback methods
        poster_url = get_movie_poster(movie_id, title)

        # Get cast photos with API fallback and Wikipedia summaries
        detailed_cast = get_cast_photos(movie_id, movie_data['detailed_cast'])

        # Prepare searched movie data
        searched_movie = {
            'title': movie_data['title_y'],
            'poster': poster_url,
            'overview': movie_data['overview'],
            'vote_average': float(movie_data['vote_average']),
            'runtime': format_runtime(movie_data['runtime']),
            'release_date': movie_data['release_date'],
            'status': movie_data.get('status', 'N/A'),
            'genres': movie_data['genres_list'],
            'tmdb_id': movie_id,
            'detailed_cast': detailed_cast
        }

        # Get recommendations
        recommendations = get_recommendations(title)
        if recommendations is None:
            return render_template('index.html', error="No recommendations found")

        # Prepare recommended movies data with posters
        recommended_movies = []
        for _, row in recommendations.iterrows():
            poster = get_movie_poster(row['id'], row['title_y'])
            recommended_movies.append({
                'title': row['title_y'],
                'poster': poster,
                'release_date': row['release_date'],
                'tmdb_id': row['id']
            })

        return render_template('recommend.html',
                               searched_movie=searched_movie,
                               recommended_movies=recommended_movies)

    except IndexError:
        return render_template('index.html', error="Movie not found in our database")
    except Exception as e:
        print(f"Error in recommendation: {e}")
        return render_template('index.html', error="Error processing your request")


@app.route('/cast/<name>')
def cast_details(name):
    summary = get_wikipedia_summary(name)
    return render_template('cast_details.html', name=name, summary=summary)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')



if __name__ == '__main__':
    app.run(debug=True)