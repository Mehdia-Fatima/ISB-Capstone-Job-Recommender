# recommender.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from functools import lru_cache

# Load jobs data
jobs_df = pd.read_csv("jobs.csv")
jobs_df['avg_salary'] = (jobs_df['Min salary'] + jobs_df['Max salary']) / 2

# Encode job skills
model = SentenceTransformer('all-MiniLM-L6-v2')
job_embeddings = model.encode(jobs_df['Job type'].tolist(), convert_to_tensor=True)

# Salary scaler
salary_scaler = MinMaxScaler()
jobs_df['salary_scaled'] = salary_scaler.fit_transform(jobs_df[['avg_salary']])

# Location
geolocator = Nominatim(user_agent="job_recommender_app")

@lru_cache(maxsize=None)
def get_coordinates(location):
    try:
        location_info = geolocator.geocode(location)
        if location_info:
            return location_info.latitude, location_info.longitude
        else:
            return None
    except GeocoderTimedOut:
        return get_coordinates(location)
        
# Precompute coordinates for unique locations
location_coords = {}

def get_location_coords(location):
    if location not in location_coords:
        coords = get_coordinates(location)
        location_coords[location] = coords
    return location_coords[location]

def calculate_location_proximity(emp_location, job_location):
    emp_coords = get_location_coords(emp_location)
    job_coords = get_location_coords(job_location)
    
    if not emp_coords or not job_coords:
        return 0

    distance = geodesic(emp_coords, job_coords).kilometers
    max_distance = 500  # Normalization threshold
    proximity_score = 1 - (distance / max_distance)
    return max(proximity_score, 0)

def recommend_jobs(user_name, user_age, user_location, user_skills, expected_salary, top_n=10):
    emp_skill_emb = model.encode([user_skills])[0]

    # Skill similarity
    skill_sim = cosine_similarity(
        [emp_skill_emb],
        [emb.cpu().numpy() for emb in job_embeddings])[0]

    # Location proximity
    location_scores = jobs_df['State'].apply(lambda x: calculate_location_proximity(user_location, x))

    # Salary similarity
    expected_salary_scaled = salary_scaler.transform([[expected_salary]])[0][0]
    salary_scores = 1 - abs(jobs_df['salary_scaled'] - expected_salary_scaled)

    # Final score
    final_scores = (
        0.5 * skill_sim +
        0.3 * location_scores +
        0.2 * salary_scores
    )

    top_idx = final_scores.argsort()[::-1][:top_n]
    recommendations = jobs_df.iloc[top_idx].copy()
    recommendations['match_score'] = final_scores[top_idx]

    return recommendations
