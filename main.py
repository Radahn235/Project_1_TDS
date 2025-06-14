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

app = FastAPI()

# Load embedded data from JSON
embedded_data = []
try:
    with open("data/embedded.json", "r") as f:
        embedded_data = json.load(f)
except FileNotFoundError:
    print("⚠️ data/embedded.json not found. Make sure to upload it before deployment.")

class QuestionRequest(BaseModel):
    question: str
    image_base64: Optional[str] = None

@app.get("/")
def read_root():
    return {"message": "Virtual TA is running!"}

@app.post("/api/")
async def answer_question(request: QuestionRequest):
    query = request.question

    # Dummy embedding - replace this with OpenAI embedding call
    query_embedding = np.random.rand(100)  # Example only

    # Convert list of lists to np.array
    embedded_vectors = np.array([item["embedding"] for item in embedded_data])
    similarities = cosine_similarity([query_embedding], embedded_vectors)[0]

    top_indices = np.argsort(similarities)[::-1][:3]
    top_links = [embedded_data[i]["link"] for i in top_indices]

    return {
        "answer": "This is a placeholder response. RAG answer can be integrated.",
        "relevant_links": top_links
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
