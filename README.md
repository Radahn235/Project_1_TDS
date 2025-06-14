# Project_1_TDS# Virtual Teaching Assistant – TDS Jan 2025

An API that answers student queries about the "Tools in Data Science" course using OpenAI and embeddings-based search on course + Discourse content.

## 🔧 Tech Stack
- FastAPI
- OpenAI GPT + embeddings
- Cosine similarity
- Python
- Railway (for deployment)

## 📌 How to Use
1. Clone the repo
2. Create `.env` with your OpenAI key
3. Run `uvicorn main:app --reload`

## 🚀 Endpoints
- `POST /api/` – Accepts questions and optional images, returns answers with relevant links.

## 📁 Data
- `data/embedded.json` – Vectorized discourse posts (100-dim).

## 🧠 Author
Priyansh Nigam – TDS Jan 2025
