import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    raise ValueError("AIPROXY_TOKEN not found in environment variables")

# Load raw discourse posts from JSON
with open("data/discourse.json", "r", encoding="utf-8") as f:

    posts = json.load(f)

# Generate embeddings
embedded_posts = []
for post in posts:
    response = requests.post(
        "https://aiproxy.sanand.workers.dev/openai/v1/embeddings",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AIPROXY_TOKEN}"
        },
        json={
            "model": "text-embedding-3-small",
            "input": [post["content"]]
        }
    )
    if response.status_code != 200:
        raise Exception(f"Failed for post {post['link']}: {response.text}")
    
    embedding = response.json()["data"][0]["embedding"]
    embedded_posts.append({
        "link": post["link"],
        "embedding": embedding
    })

# Save to embedded.json
with open("data/embedded.json", "w", encoding="utf-8") as f:
    json.dump(embedded_posts, f)

print("✅ Embeddings regenerated and saved to data/embedded.json")
