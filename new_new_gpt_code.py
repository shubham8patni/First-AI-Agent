from flask import Flask, request, jsonify
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from faiss import IndexFlatL2  # ✅ Import directly from FAISS
import faiss
import numpy as np
import os
import fitz  # PyMuPDF
import re
import json

# Load insurance policy JSON
def load_policy_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Convert JSON data into plain text for embedding
def format_policy_data(policy_json):
    formatted_texts = []
    for section, details in policy_json.items():
        formatted_texts.append(f"Section: {section}\nDetails: {json.dumps(details, indent=2)}")
    return "\n".join(formatted_texts)

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS vector store for efficient search
def create_vector_store(policy_texts):
    index = faiss.IndexFlatL2(384)  # ✅ 384D index for MiniLM embeddings
    index = faiss.IndexIDMap(index)  # ✅ Wrap in IndexIDMap to allow custom IDs

    vectors = np.array([embedding_model.embed_query(text) for text in policy_texts], dtype=np.float32)
    ids = np.arange(len(vectors), dtype=np.int64)  # Generate sequential IDs

    index.add_with_ids(vectors, ids)  # ✅ Now adding vectors with IDs works!
    
    return index


# Initialize LLM (replace with LLaMA 3 or OpenAI's GPT-based model)
# llm = OpenAI(model="gpt-4")
llm =  HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-70B",#"mistralai/Mistral-7B-Instruct",#"meta-llama/Meta-Llama-3-70B",
    temperature=0.3,
    top_p=0.6,
    task="text-generation",
    model_kwargs={"max_length": 4096}  # Increase for multi-step answers
)

# Load and process policies
policy_json = load_policy_data("zurich_insurance_jon.json")
formatted_policy_texts = format_policy_data(policy_json).split("\n")
vector_store = create_vector_store(formatted_policy_texts)

# Build Retrieval Engine
def retrieve_relevant_policies(query):
    query_vector = embedding_model.embed_query(query)  # Generates a list
    query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)  # ✅ Convert to NumPy array

    _, indices = vector_store.search(query_vector, k=3)  # ✅ Now works!
    
    return [formatted_policy_texts[i] for i in indices[0]]

# Query LLM for response
def generate_response(query):
    relevant_policies = retrieve_relevant_policies(query)
    context = "\n".join(relevant_policies)
    prompt = f"User Query: {query}\n\nRelevant Policy Sections:\n{context}\n\nAnswer the query based on these policies."
    # prompt = f"Answer the following insurance-related question: {user_query}"
    
    response = llm.invoke(prompt)  # ✅ Use `.invoke()` instead of `.complete()`
    
    return response

# User interaction
while True:
    user_query = input("\nEnter your insurance-related question (or 'exit' to quit): ")
    if user_query.lower() == "exit":
        break
    response = generate_response(user_query)
    print("\nAI Response:\n", response)



# //Does my winter sports coverage include snowboarding?