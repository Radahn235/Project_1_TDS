import requests
import json

BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"
CATEGORY_ID = 123  # Replace with actual

def fetch_posts():
    url = f"{BASE_URL}/c/tds/Quiz-1.json"
    response = requests.get(url)
    data = response.json()
    posts = []

    for topic in data.get("topic_list", {}).get("topics", []):
        posts.append({
            "id": topic["id"],
            "title": topic["title"],
            "url": f"{BASE_URL}/t/{topic['slug']}/{topic['id']}"
        })

    with open("data/discourse.json", "w") as f:
        json.dump(posts, f, indent=2)

if __name__ == "__main__":
    fetch_posts()
