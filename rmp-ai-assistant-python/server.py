from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv()
import os
import json
import torch
import requests
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from huggingface_hub import login
login(os.getenv("HUGGINGFACE_TOKEN"))

# App Instance 
app = Flask(__name__)
CORS(app)

# AI Models Config
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
# genai_config = genai.GenerationConfig(
#     response_schema='application/json'
# )
chat_model = genai.GenerativeModel(
    model_name='gemini-1.5-flash'
)
embedding_model = SentenceTransformer("TaylorAI/gte-tiny")
systemPrompt = """
You are a rate my professor agent to help students find classes, that takes in user questions and answers them. 
For every user question, the top 5 professors that match the user's question will be added under it. 
Use them to form a summary IN YOUR OWN WORDS where you zero down to 3 professors that you think will be the best choice for the user.
Put 2 newlines between the possible professors.
"""

# Pinecone Config
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index('rag')

@app.route("/api/recommendation", methods=["POST"])
def recommendation():

    request_data = request.get_json()
    user_input_embeddings = embedding_model.encode(request_data["content"])
    
    query_results = index.query(
        namespace="ns1",
        vector=user_input_embeddings.tolist(),
        top_k=5,
        include_metadata=True
    )

    resultString = ''
    for match in query_results.matches:
        resultString += "Professor: " + match.id + ", Review: " + match.metadata['review'] + ", Subject: " + match.metadata['subject'] + ", Stars: " + str(match.metadata['stars']) + "\n\n" 

    content = [{"role": "model", "parts": [{"text": systemPrompt}]}, 
               {"role": "user", "parts": [{"text": request_data["content"] + "\n\n" + resultString}]}]
    response = chat_model.generate_content(content)

    # print(response)

    return jsonify({"response": response.text})

if __name__ == "__main__":
    app.run(debug=True, port=8080)