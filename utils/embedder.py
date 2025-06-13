import openai
import json

openai.api_key = "YOUR_OPENAI_API_KEY"

def embed_posts():
    with open("data/discourse.json", "r") as f:
        posts = json.load(f)

    for post in posts:
        post["embedding"] = openai.embeddings.create(
            input=post["title"],
            model="text-embedding-3-small"
        )["data"][0]["embedding"]

    with open("data/embedded.json", "w") as f:
        json.dump(posts, f, indent=2)

if __name__ == "__main__":
    embed_posts()
