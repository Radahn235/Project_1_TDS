from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import json
import numpy as np
import requests
from scipy.spatial.distance import cosine

# FastAPI app
app = FastAPI()

# Load embedded Discourse data
try:
    with open("data/embedded.json", "r", encoding="utf-8") as f:
        embedded_data = json.load(f)
except Exception as e:
    raise RuntimeError(f"❌ Could not load embedded.json: {e}")

# Get AIPROXY token
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    raise ValueError("❌ AIPROXY_TOKEN not found in .env file")

# Define request body
class Question(BaseModel):
    question: str

# Helper function to get embedding from AIProxy
def get_embedding(text: str):
    response = requests.post(
        "https://aiproxy.sanand.workers.dev/openai/v1/embeddings",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AIPROXY_TOKEN}"
        },
        json={
            "model": "text-embedding-3-small",
            "input": [text]
        }
    )
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Embedding error: {response.text}")
    return response.json()["data"][0]["embedding"]

# Find best match using cosine similarity
def find_best_match(question_vector):
    best_score = -1
    best_link = None
    for item in embedded_data:
        vector = item["embedding"]
        similarity = 1 - cosine(question_vector, vector)
        if similarity > best_score:
            best_score = similarity
            best_link = item["link"]
    return best_link, best_score

# FastAPI route
@app.post("/api/")
def answer_question(q: Question):
    try:
        question_vector = get_embedding(q.question)
        link, score = find_best_match(question_vector)
        return {
            "answer": f"This might help: {link}",
            "relevance": round(score, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/")
def root():
    return {"message": "Virtual TA is running!"}
