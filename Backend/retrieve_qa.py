import os
import json
import requests
from flask import Flask, request, jsonify
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from flask_cors import CORS

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise ValueError("❌ API key not found in .env file!")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow all origins
# Embedding model and vector store setup
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory="./grownius_vdb", embedding_function=embedding_model)

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def retrieve_context(query, top_k=3):
    """Retrieves relevant knowledge from ChromaDB."""
    results = vector_store.similarity_search(query, k=top_k)
    return "\n\n".join([doc.page_content for doc in results]) if results else "No relevant information found."

@app.route("/chat", methods=["POST"])
def chat():
    """Handles chat requests from frontend."""
    print("Chat request received!")
    data = request.json
    
    user_query = data.get("prompt")
    
    if not user_query:
        return jsonify({"error": "User query is required!"}), 400
    
    # Retrieve knowledge
    context = retrieve_context(user_query)
    
    
    # Modify prompt to act as an agriculture expert
    modified_prompt = f"""
    Act like an agriculture expert and AI assistant for me. Give me only suitable information.
    User Query: {user_query}
    
    ### Retrieved Knowledge:  
    {context}
    """
    
    # OpenRouter API request
    request_payload = {
        "model": "openai/gpt-4o",
        "messages": [{"role": "user", "content": modified_prompt}]
    }
    
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=request_payload)
    
    if response.status_code == 200:
        ai_response = response.json()["choices"][0]["message"]["content"]
        return jsonify({"rag_response": ai_response})
    else:
        return jsonify({"error": "AI API Error", "details": response.text}), response.status_code

if __name__ == "__main__":
    app.run(debug=True,port=5002)
