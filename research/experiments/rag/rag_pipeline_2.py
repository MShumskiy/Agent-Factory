import os
import psycopg2 as psql
from pydantic import BaseModel
from typing import  List
import sys
import os
import numpy as np
from sentence_transformers import SentenceTransformer,CrossEncoder
import json

# Add the project root to sys.path to import from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.llmp_utils import llmp_call

import gc
import torch

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RagPipeline():
    
    def __init__(self, DB_CONFIG, rag_configs):

        self.DB_CONFIG = DB_CONFIG
        self.system_prompt = rag_configs['system_prompt_rag']
        self.embeddings_model_id = rag_configs['embeddings_model_id']
        self.cross_encoder_id = rag_configs['cross_encoder_id']
        self.top_k = rag_configs['top_k']
        self.temperature = rag_configs['temperature']
        self.ce_threshold = rag_configs['ce_threshold']
        self.search_type = rag_configs['search_type']
        self.src = rag_configs['src']
        self.embed_table = rag_configs["embed_table"]

        
        
    def retrieve_embeddings(self, selected_kb):
        """
        Retrieves all embeddings from the database.
        """


        conn = psql.connect(**self.DB_CONFIG)
        cur = conn.cursor()
        
        # Fetch all embeddings from the database
        query = f"""
            SELECT id, text, pages, token_count, embedding, embeddings_model, document
            FROM {self.embed_table}
            WHERE kb = %s;
        """
        cur.execute(query, (selected_kb,))
        results = cur.fetchall()
        cur.close()
        conn.close()
        
        return results


    def convert_chunk_to_json(self, embedding_tuple):
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

    def semantic_search(self,user_input,db_embeddings,embeddings_model):
        """
        Performs semantic search on the database of embeddings.
        """
        # Encode user input
        user_embedding = embeddings_model.encode(user_input)
        
        # Calculate cosine similarity between user input and all embeddings
        similarities = []
        for db_embedding in db_embeddings:
            db_embedding_vector = np.array(db_embedding[4])
            if self.search_type == "cosine":
                similarity = np.dot(user_embedding, db_embedding_vector) / (np.linalg.norm(user_embedding) * np.linalg.norm(db_embedding_vector))
            elif self.search_type == "euclidean":
                similarity = np.linalg.norm(user_embedding - db_embedding_vector)
            elif self.search_type == "dot":
                similarity = np.dot(user_embedding, db_embedding_vector) / (np.linalg.norm(user_embedding) * np.linalg.norm(db_embedding_vector))
            elif self.search_type == "manhattan":
                similarity = np.linalg.norm(user_embedding - db_embedding_vector, ord=1)
            elif self.search_type == "minkowski":
                similarity = np.linalg.norm(user_embedding - db_embedding_vector, ord=2)
            similarities.append((db_embedding, similarity))
            
            
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        similarities = [self.convert_chunk_to_json(similarity) for similarity in similarities]
        #similarities = process_similarities(similarities)
        
        # Return top_p results
        return similarities[:self.top_k]

    

    def process_context(self, user_input, selected_chunks, cross_encoder):
        """
        Processes context to be fed into the LLM.
        """
        results = cross_encoder.predict([[user_input, chunk['text']] for chunk in selected_chunks])/200
        softmax_scores = softmax(results)

        for chunk, score in zip(selected_chunks, softmax_scores):
            chunk['ce_score'] = score
            
        filtered_chunks = [chunk for chunk in selected_chunks if chunk['ce_score'] > self.ce_threshold]

        context = " ".join([chunk['text'] for chunk in filtered_chunks])
        
        return context,filtered_chunks

    def get_references(self,selected_chunks):
        document_pages = {}

        for chunk in selected_chunks:
            doc = chunk['document']
            if doc not in document_pages:
                document_pages[doc] = set()  # Use a set to ensure uniqueness

            document_pages[doc].update(chunk["pages"])

        # Convert sets to sorted lists for better readability
        document_pages = {doc: sorted(pages) for doc, pages in document_pages.items()}

        return document_pages

    def generate_rag(self, user_prompt, selected_kb,llm):
        """
        Generates a response using the RAG pipeline.
        """

        
        output_format = None
        print(self.embeddings_model_id)

        print("Retrieving embeddings...")
        db_embeddings = self.retrieve_embeddings(selected_kb)
        embeddings_model = SentenceTransformer(self.embeddings_model_id)
        print("Performing semantic search...")
        selected_chunks = self.semantic_search(user_prompt, db_embeddings,embeddings_model)

        print("Unloading embeddings model...")
        del embeddings_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        #return selected_chunks      
        print("Processing context...")
        cross_encoder = CrossEncoder(self.cross_encoder_id)
        context, filtered_chunks = self.process_context(user_prompt, selected_chunks, cross_encoder)
        del cross_encoder
        print("Processing references...")
        document_pages = self.get_references(selected_chunks)
        prompt = f"Based only on the following in markdown: {context} \nAnswer this, without hallucinating or making information up: {user_prompt}"

        print("Calling LLMP...")
        # Call llmp_call function that was imported at the top
        
        response = llmp_call(prompt, self.system_prompt, llm, self.temperature, self.src, output_format)
        return response,document_pages,selected_chunks,context


# HELPER CLASSES
##################################################################################
class EmbeddingChunk(BaseModel):
    id:int
    text:str
    pages:List[int]
    token_count:int
    embedding:List[float]
    document:str
    similarity:float
        
        
##################################################################################

# HELPER FUNCTIONS
##################################################################################
def softmax(x):
        """Computes softmax for an array of scores."""
        exp_x = np.exp(x - np.max(x))  # Subtract max score for numerical stability
        return exp_x / exp_x.sum()
##################################################################################