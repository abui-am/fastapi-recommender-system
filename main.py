from typing import Union

from fastapi import FastAPI

import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from json import loads

from dotenv import load_dotenv

load_dotenv()

import os

app = FastAPI()


def recommend_products(target_product_id):
    beUrl =  os.getenv('NEXTJS_URL')

    print(beUrl)
    # Load data into a DataFrame
    all_brands = requests.get( beUrl +"/api/brands?unmarked_only=true")
    print(all_brands.json())
    brands_df=pd.json_normalize(all_brands.json())
    
    # Retrieve target product details
    resp_target_product = requests.get( beUrl +"/api/brands/"+str(target_product_id))
    target_product_json = resp_target_product.json()
    
    # Separate features for similarity calculations
    target_price = target_product_json['price']
    target_name = target_product_json['name']
    target_tags = target_product_json['tag']

    # Price similarity (normalized difference)
    brands_df['price_similarity'] = 1 - abs(brands_df['price'] - target_price) / (brands_df['price'].max() - brands_df['price'].min())

    # Name similarity using TF-IDF and cosine similarity
    tfidf_vectorizer = TfidfVectorizer()
    name_vectors = tfidf_vectorizer.fit_transform(brands_df['name'])
    target_name_vector = tfidf_vectorizer.transform([target_name])
    brands_df['name_similarity'] = cosine_similarity(name_vectors, target_name_vector).flatten()

    # Tag similarity: count matching tags and normalize
    tag_vectors = tfidf_vectorizer.fit_transform(brands_df['tag'])
    target_tags_vector = tfidf_vectorizer.transform([target_tags])
    brands_df['tag_similarity'] =  cosine_similarity(tag_vectors,target_tags_vector).flatten()
    
    # Compute overall similarity score with weights
    brands_df['similarity_score'] = (brands_df['price_similarity'] * 0.3 +
                                     brands_df['name_similarity'] * 0.3 +
                                     brands_df['tag_similarity'] * 0.4) 
    
    # If brand_df['boosted'] == True, boost the score by 1.5
    brands_df.loc[brands_df['boosted'] == True, 'similarity_score'] *= 1.5

    recommendations = brands_df[brands_df['id'] != target_product_id].sort_values(by='similarity_score', ascending=False)
    
    # Output the top recommendations
    print("Top Recommendations:")
    for idx, row in recommendations.head(6).iterrows():  # Adjust head(n) for more or fewer recommendations
        print(f"Product ID: {row['id']}, Name: {row['name']}, Price: {row['price']}, Similarity Score: {row['similarity_score']:.2f}")

    return recommendations.head(6)  # Return top 5 recommendations


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/api/brands/{product_id}")
def recommend_products_api(product_id: str):
    beUrl =  os.getenv('NEXTJS_URL')
    resp_target_product = requests.get( beUrl +"/api/brands/"+str(product_id))
    target_product_json = resp_target_product.json()
    rec = recommend_products(product_id)
    response_data = {
        "target" : target_product_json,
        "top_4" : loads(rec.to_json(orient='records'))
    }
    return response_data