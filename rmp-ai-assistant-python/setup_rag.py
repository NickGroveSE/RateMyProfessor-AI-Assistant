from dotenv import load_dotenv
load_dotenv()
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from huggingface_hub import InferenceClient
import os
import json
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
login(os.getenv("HUGGINGFACE_TOKEN"))

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create a Pinecone index (Commented out since we do not need to run this multiple times)
# pc.create_index(
#     name="rag",
#     dimension=384,
#     metric="cosine",
#     spec=ServerlessSpec(cloud="aws", region="us-east-1"),
# )

data = json.load(open("reviews.json"))

processed_data = []

model = SentenceTransformer("TaylorAI/gte-tiny")
embeddings = model.encode(data["reviews"])

# Create embeddings for each review
for index, review in enumerate(data["reviews"]):
    # embedding = client.feature_extraction(
    #     text=review['review'], model="Xenova/all-MiniLM-L6-v2"
    # )

    processed_data.append(
        {
            "values": embeddings[index],
            "id": review["professor"],
            "metadata":{
                "review": review["review"],
                "subject": review["subject"],
                "stars": review["stars"],
            }
        }
    )

# Insert the embeddings into the Pinecone index
index = pc.Index("rag")
upsert_response = index.upsert(
    vectors=processed_data,
    namespace="ns1",
)
print(f"Upserted count: {upsert_response['upserted_count']}")

# Print index statistics
print(index.describe_index_stats())


