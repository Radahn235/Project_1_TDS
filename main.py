from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import base64
from io import BytesIO
from PIL import Image
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables (for OPENAI_API_KEY)
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Load embedded data
with open("data/embedded.json", "r") as f:
    embedded_data = json.load(f)

embedded_texts = [item["text"] for item in embedded_data]
embedded_vectors = np.array([item["embedding"] for item in embedded_data])

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Request schema
class QuestionRequest(BaseModel):
    question: str
    image: Optional[str] = None  # base64 string

# Embedding function
def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Answering endpoint
@app.post("/api/")
async def answer_question(request: QuestionRequest):
    # Get embedding for user question
    query_embedding = get_embedding(request.question)

    # Check dimensional match
    if len(query_embedding) != embedded_vectors.shape[1]:
        return {
            "error": f"Embedding size mismatch: query has {len(query_embedding)}, but database has {embedded_vectors.shape[1]}"
        }

    # Compute cosine similarity
    similarities = cosine_similarity([query_embedding], embedded_vectors)[0]

    # Top 3 similar texts
    top_indices = similarities.argsort()[-3:][::-1]
    top_texts = [embedded_texts[i] for i in top_indices]

    return {
        "answer": "This is a relevant response based on your question.",
        "related_posts": top_texts,
        "similarities": [float(similarities[i]) for i in top_indices]
    }

# Health check
@app.get("/")
def health_check():
    return {"message": "Virtual TA is running!"}
