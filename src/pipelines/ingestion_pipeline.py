
from PyPDF2 import PdfReader
from pydantic import BaseModel
from typing import List
import sys
import os
from sentence_transformers import SentenceTransformer
import json

import psycopg2 as psql

import gc
import torch

from dotenv import load_dotenv

load_dotenv()


# project_root = os.path.abspath("..")

# if project_root not in sys.path:
#     sys.path.append(project_root)

docs_path = "../data/documents/"
documents_titles = os.listdir(docs_path)

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
config_file_path = os.path.join(root_path_dir, 'configs', 'ingestion_configs.json')

        
with open(config_file_path, 'r') as config_file:
        config = json.load(config_file)
        
chunk_size = config['chunk_size']
embeddings_model_id = config['embeddings_model_id']

class DocumentObject(BaseModel):
    document: str
    pages: list

class PageObject(BaseModel):
    page_number: int
    text: str
    size: int
    document: str
    
class Chunk(BaseModel):
    text: str
    pages: List[int]
    token_count: int
    document: str





def process_document(document_title):
    
    """Processes document and returns a DocumentObject with all its pages processed.

    Args:
        document_title (str): document file name

    Returns:
        DocumentObject: Document object
    """
    
    document_path = os.path.join(docs_path, documents_titles[0])
    reader = PdfReader(document_path)
    
    page_objects = []
    
    # process each page to generate PageObject
    for i, page in enumerate(reader.pages):
        
        page_text = page.extract_text()
        page_size = int(len(page_text)/4.7)
        page_object = PageObject(page_number=i+1, text=page_text, size=page_size, document=document_title)
        
        page_objects.append(page_object.model_dump())
    
    # creates DocumentObject with PageObjects' contents
    document_object = DocumentObject(document=document_title, pages=page_objects)
    
    return document_object


def create_chunks(document_object, chunk_size):
    """
    Creates fixed-size text chunks from a DocumentObject. If a page has fewer tokens than chunk_size,
    tokens from subsequent pages are taken to fill the chunk. In cases where a single page contributes
    tokens to a chunk, the 'pages' field will contain a list of one page number; if tokens come from
    multiple pages, that list will include all relevant page numbers.
    
    Args:
        document_object (DocumentObject): The processed document containing pages.
        chunk_size (int): Desired number of tokens per chunk.
        
    Returns:
        list: A list of dictionaries, each with keys:
            - "chunk_text": A string containing exactly chunk_size tokens.
            - "pages": A list of page numbers that contributed tokens for this chunk.
    """
    # Build a combined list of (token, page_number) tuples
    tokens_with_page = []
    for page in document_object.pages:
        # Split text into tokens (using default whitespace splitting)
        tokens = page["text"].split()
        # Add each token with its source page number
        tokens_with_page.extend([(token, page["page_number"]) for token in tokens])
    
    chunks = []
    total_tokens = len(tokens_with_page)
    
    # Iterate over tokens_with_page in steps of chunk_size
    # This ensures each chunk is exactly chunk_size tokens
    for i in range(0, total_tokens - chunk_size + 1, chunk_size):
        chunk_slice = tokens_with_page[i:i+chunk_size]
        # Reconstruct the chunk text from tokens
        chunk_text = " ".join(token for token, _ in chunk_slice)
        # Collect the page numbers (maintaining order and uniqueness)
        pages = []
        for _, page_number in chunk_slice:
            if page_number not in pages:
                pages.append(page_number)
                
                
        chunk = Chunk(
            text=chunk_text,
            pages=pages,
            token_count=len(chunk_slice),
            document=document_object.document
        )
        chunks.append(chunk.model_dump())
        
    return chunks


def create_embeddings(chunks,embeddings_model):
    """
    Creates embeddings for each chunk in a list of chunks using a SentenceTransformer model.
    """
    
    for chunk in chunks:
        embedding = embeddings_model.encode(chunk["text"])
        chunk["embedding"] = embedding
        
    return chunks

        
        
def ensure_table_exists():
    """
    Creates the document_chunks table if it doesn't exist.
    """
    
    create_table_query = """
    CREATE TABLE IF NOT EXISTS embeddings_table_v2 (
        id SERIAL PRIMARY KEY,
        document TEXT NOT NULL,
        text TEXT NOT NULL,
        pages INTEGER[],
        token_count INTEGER,
        embedding JSONB,
        embeddings_model TEXT,
        UNIQUE (text, document, embeddings_model)
    );
    """
    conn = psql.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(create_table_query)
    conn.commit()
    cur.close()
    conn.close()
    
    
    

def save_embeddings(embeddings,embeddings_model_id):
    """
    Saves a list of chunks with embeddings into PostgreSQL.
    """
    ensure_table_exists()  # Ensure table exists before inserting

    conn = psql.connect(**DB_CONFIG)
    cur = conn.cursor()

    insert_query = """
    INSERT INTO embeddings_table_v2 (text, pages, token_count, document, embedding, embeddings_model)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (text, document, embeddings_model) DO NOTHING;  -- Skips duplicates
    """

    for embedding in embeddings:
        # Convert NumPy array to Python list for JSON storage
        embedding_vector = embedding["embedding"].tolist()
        try:
            cur.execute(insert_query, (
                    embedding["text"],
                    embedding["pages"],
                    embedding["token_count"],
                    embedding["document"],
                    json.dumps(embedding_vector),
                    embeddings_model_id# Convert list to JSONB
                ))
        except:
            continue

    conn.commit()
    cur.close()
    conn.close()
    
def get_unique_documents():
    """
    Returns a list of unique document titles from the database.
    """
    conn = psql.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT document, embeddings_model, token_count FROM embeddings_table_v2 GROUP BY document, embeddings_model, token_count;")
    documents = [row for row in cur.fetchall()]
    cur.close()
    conn.close()
    return documents
    
    
def ingest_document(chunk_size,document_title,embeddings_model_id):
    """
    Ingests a document into the database by processing it, creating chunks, and saving embeddings.
    """
    
    print(f"Processing document: {document_title}")
    document_object = process_document(document_title)
    
    print("Creating chunks...")
    chunks = create_chunks(document_object, chunk_size)

    print("loading embeddings model...")
    embeddings_model = SentenceTransformer(embeddings_model_id)
    
    print("Creating embeddings...")
    embeddings = create_embeddings(chunks,embeddings_model)

    print("Unloading embeddings model...")
    del embeddings_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
    print("Checking table...")
    ensure_table_exists()
    
    print("Saving embeddings...")
    save_embeddings(embeddings,embeddings_model_id)
    
    print(f"{document_title} has been ingested.")
    

def ingest_pipeline():
    """
    Ingests all documents in the data/documents folder into the database. If a document has already been ingested,
    """

    unique_configs = get_unique_documents()
    documents_titles = os.listdir(docs_path)

    for document in documents_titles:
        
        config = (document,embeddings_model_id,chunk_size)
        
        if config not in unique_configs:
            ingest_document(chunk_size,document,embeddings_model_id)
        else: 
            print(f"{document} already ingested with the same configuration.")