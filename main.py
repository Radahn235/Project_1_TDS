import os
import json
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
import requests

# Load environment variable
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    raise RuntimeError("Missing AIPROXY_TOKEN in environment variables.")

# Load embedded data
with open("data/embedded.json", "r") as f:
    embedded_data = json.load(f)

# Convert embeddings to numpy arrays
for item in embedded_data:
    item["embedding"] = np.array(item["embedding"])

# FastAPI app
app = FastAPI()

# Request schema
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3

# Compute cosine similarity
def cosine_similarity(vec1, vec2):
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Get embedding for a new query
def get_embedding_from_api(text: str):
    url = "https://openrouter.ai/api/v1/embeddings"  # or your proxy
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "text-embedding-3-small",
        "input": text
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch embedding")
    return np.array(response.json()["data"][0]["embedding"])

# Main API
@app.post("/api/")
def answer_question(req: QueryRequest):
    query_embedding = get_embedding_from_api(req.query)

    # Compute cosine similarities
    similarities = []
    for item in embedded_data:
        sim = cosine_similarity(query_embedding, item["embedding"])
        similarities.append((sim, item))

    # Sort and return top K
    similarities.sort(reverse=True, key=lambda x: x[0])
    top_results = similarities[:req.top_k]

    return {
        "query": req.query,
        "top_matches": [
            {
                "text": item["text"],
                "url": item.get("url", ""),
                "score": round(score, 4)
            }
            for score, item in top_results
        ]
    }

# Health check
@app.get("/")
def root():
    return {"message": "Virtual TA API is running."}
