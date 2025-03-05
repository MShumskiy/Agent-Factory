import os
import psycopg2 as psql
from pydantic import BaseModel
from typing import  List
import sys
import os
import numpy as np
from sentence_transformers import SentenceTransformer,CrossEncoder
from utils.llmp_utils import llmp_call
import json

import gc
import torch

db_database = os.getenv("DB_DATABASE")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")

DB_CONFIG = {
    "dbname": db_database,
    "user": db_user,
    "password": db_password,
    "host": db_host
}

root_path = ".."
root_path_dir = os.path.abspath(root_path)
config_file_path = os.path.join(root_path_dir, 'configs', 'rag_configs.json')

        
with open(config_file_path, 'r') as config_file:
        config = json.load(config_file)
        
system_prompt = config['system_prompt_rag']
embeddings_model_id = config['embeddings_model_id']
cross_encoder_id = config['cross_encoder_id']
top_k = config['top_k']

class EmbeddingChunk(BaseModel):
    id:int
    text:str
    pages:List[int]
    token_count:int
    embedding:List[float]
    document:str
    similarity:float


def retrieve_embeddings():
    """
    Retrieves all embeddings from the database.
    """
    
    
    conn = psql.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Fetch all embeddings from the database
    cur.execute("SELECT id, text, pages, token_count, embedding, embeddings_model, document FROM embeddings_table_v2;")
    results = cur.fetchall()
    cur.close()
    conn.close()
    
    return results


def convert_chunk_to_json(embedding_tuple):
    """
    Converts a chunk of embeddings to a JSON object.
    """
    
    return EmbeddingChunk(
        id=embedding_tuple[0][0],
        text=embedding_tuple[0][1],
        pages=embedding_tuple[0][2],
        token_count=embedding_tuple[0][3],
        embedding=embedding_tuple[0][4],
        embeddings_model=embedding_tuple[0][5],
        document=embedding_tuple[0][6],
        similarity=embedding_tuple[1]
    ).model_dump()
    
    
def semantic_search(user_input,db_embeddings, top_k, model):
    """
    Performs semantic search on the database of embeddings.
    """
    
    user_embedding = model.encode(user_input)
    
    # Calculate cosine similarity between user input and all embeddings
    similarities = []
    for db_embedding in db_embeddings:
        db_embedding_vector = np.array(db_embedding[4])
        similarity = np.dot(user_embedding, db_embedding_vector) / (np.linalg.norm(user_embedding) * np.linalg.norm(db_embedding_vector))
        similarities.append((db_embedding, similarity))
        
        
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    similarities = [convert_chunk_to_json(similarity) for similarity in similarities]
    #similarities = process_similarities(similarities)
    
    # Return top_p results
    return similarities[:top_k]

def softmax(x):
    """Computes softmax for an array of scores."""
    exp_x = np.exp(x - np.max(x))  # Subtract max score for numerical stability
    return exp_x / exp_x.sum()

def process_context(user_input,selected_chunks,cross_encoder):
    """
    Processes context to be fed into the LLM.
    """
    results = cross_encoder.predict([[user_input, chunk['text']] for chunk in selected_chunks])/200
    softmax_scores = softmax(results)

    threshold = 0.01

    filtered_chunks = [selected_chunks[i] for i, score in enumerate(softmax_scores) if score > threshold]

    context = " ".join([chunk['text'] for chunk in filtered_chunks])
    
    return context

def get_references(selected_chunks):
    document_pages = {}

    for chunk in selected_chunks:
        doc = chunk['document']
        if doc not in document_pages:
            document_pages[doc] = set()  # Use a set to ensure uniqueness

        document_pages[doc].update(chunk["pages"])

    # Convert sets to sorted lists for better readability
    document_pages = {doc: sorted(pages) for doc, pages in document_pages.items()}

    return document_pages

def generate_rag(model, user_prompt):
    """
    Generates a response using the RAG pipeline.
    """
    
    print("Retrieving embeddings...")
    db_embeddings = retrieve_embeddings()
    embeddings_model = SentenceTransformer(embeddings_model_id)
    print("Performing semantic search...")
    selected_chunks = semantic_search(user_prompt, db_embeddings, top_k, embeddings_model)
    
    print("Unloading embeddings model...")
    del embeddings_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
                
    print("Processing context...")
    cross_encoder = CrossEncoder(cross_encoder_id)
    context = process_context(user_prompt,selected_chunks,cross_encoder)
    print("Processing references...")
    document_pages = get_references(selected_chunks)
    prompt = f"Based only on the following in markdown: {context} \nAnswer this: {user_prompt}"
    print("Calling LLMP...")
    response = llmp_call(prompt, system_prompt, model)
    
    return response,document_pages

