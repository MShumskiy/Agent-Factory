import os
import psycopg2 as psql
from pydantic import BaseModel
from typing import  List
import sys
import os
import numpy as np
from sentence_transformers import SentenceTransformer,CrossEncoder
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

# root_path = ".."
# root_path_dir = os.path.abspath(root_path)
# config_file_path = os.path.join(root_path_dir, 'configs', 'rag_configs.json')
config_file_path = "src/agent_factory/config/rag_configs.json"

        
with open(config_file_path, 'r') as config_file:
        config = json.load(config_file)
        
system_prompt = config['system_prompt_rag']
embeddings_model_id = config['embeddings_model_id']
cross_encoder_id = config['cross_encoder_id']
top_k = config['top_k']
temperature = config['temperature']
ce_threshold = config['ce_threshold']
search_type = config['search_type']
src = config['src']

class EmbeddingChunk(BaseModel):
    id:int
    text:str
    pages:List[int]
    token_count:int
    embedding:List[float]
    document:str
    similarity:float


def retrieve_embeddings(selected_kb):
    """
    Retrieves all embeddings from the database.
    """
    
    
    conn = psql.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Fetch all embeddings from the database
    query = """
        SELECT id, text, pages, token_count, embedding, embeddings_model, document
        FROM embeddings_table_v4
        WHERE kb = %s;
    """
    cur.execute(query, (selected_kb,))
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
    
    
def semantic_search(user_input,db_embeddings, top_k, model, search_type):
    """
    Performs semantic search on the database of embeddings.
    """
    
    user_embedding = model.encode(user_input)
    
    # Calculate cosine similarity between user input and all embeddings
    similarities = []
    for db_embedding in db_embeddings:
        db_embedding_vector = np.array(db_embedding[4])
        if search_type == "cosine":
            similarity = np.dot(user_embedding, db_embedding_vector) / (np.linalg.norm(user_embedding) * np.linalg.norm(db_embedding_vector))
        elif search_type == "euclidean":
            similarity = np.linalg.norm(user_embedding - db_embedding_vector)
        elif search_type == "dot":
            similarity = np.dot(user_embedding, db_embedding_vector) / (np.linalg.norm(user_embedding) * np.linalg.norm(db_embedding_vector))
        elif search_type == "manhattan":
            similarity = np.linalg.norm(user_embedding - db_embedding_vector, ord=1)
        elif search_type == "minkowski":
            similarity = np.linalg.norm(user_embedding - db_embedding_vector, ord=2)
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

def process_context(user_input,selected_chunks,cross_encoder,ce_threshold):
    """
    Processes context to be fed into the LLM.
    """
    results = cross_encoder.predict([[user_input, chunk['text']] for chunk in selected_chunks])/200
    softmax_scores = softmax(results)

    for chunk, score in zip(selected_chunks, softmax_scores):
        chunk['ce_score'] = score
        
    filtered_chunks = [chunk for chunk in selected_chunks if chunk['ce_score'] > ce_threshold]

    context = " ".join([chunk['text'] for chunk in filtered_chunks])
    
    return context,filtered_chunks

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

def generate_rag(model, user_prompt, selected_kb, override_config=None):
    """
    Generates a response using the RAG pipeline.
    """
    # LOAD TESTING CONFIGS
    global system_prompt,embeddings_model_id,cross_encoder_id,top_k,temperature,ce_threshold, src
    
    output_format = None
    print(embeddings_model_id)
    if override_config:
        model = override_config.get('model', model)
        embeddings_model_id = override_config.get('embeddings_model_id', embeddings_model_id)
        cross_encoder_id = override_config.get('cross_encoder_id', cross_encoder_id)
        top_k = override_config.get('top_k', top_k)
        system_prompt = override_config.get('system_prompt_rag', system_prompt)
        temperature = override_config.get('temperature', temperature)
        ce_threshold = override_config.get('ce_threshold', ce_threshold)
        src = override_config.get('src', src)
        output_format = override_config.get('format', format)

    print("Retrieving embeddings...")
    db_embeddings = retrieve_embeddings(selected_kb)
    embeddings_model = SentenceTransformer(embeddings_model_id)
    print("Performing semantic search...")
    selected_chunks = semantic_search(user_prompt, db_embeddings, top_k, embeddings_model,search_type)
    
    print("Unloading embeddings model...")
    del embeddings_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    #return selected_chunks      
    print("Processing context...")
    cross_encoder = CrossEncoder(cross_encoder_id)
    context, filtered_chunks = process_context(user_prompt,selected_chunks,cross_encoder,ce_threshold)
    print("Processing references...")
    document_pages = get_references(selected_chunks)
    prompt = f"Based only on the following in markdown: {context} \nAnswer this, without hallucinating or making information up: {user_prompt}"
    
    print("Calling LLMP...")
    # Import locally to avoid circular import
    from ..core.llmp_utils import llmp_call
    response = llmp_call(prompt, system_prompt, model, temperature, src, output_format)
    # max_retries = 3
    # for attempt in range(max_retries):
    #     try:
    #         response = llmp_call(prompt, system_prompt, model, temperature, src)
    #         break
    #     except Exception as e:
    #         print(f"Attempt {attempt + 1} failed: {e}")
            
    
    # TESTING CASE
    if override_config:
        return response,document_pages
    else:
        return response,document_pages

