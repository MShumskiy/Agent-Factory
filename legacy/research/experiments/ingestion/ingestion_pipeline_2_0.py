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

class IngPipeline():
    
    def __init__(self,DB_CONFIG,ingestion_configs):
        """
        Initializes an instance of the IngPipeline class.

        This class is used to process PDF documents and generate embeddings for their text content.

        """  
        self.DB_CONFIG = DB_CONFIG
        self.chunk_size = ingestion_configs["chunk_size"]
        self.embed_model_id = ingestion_configs["embed_model"]
        # dont forget to initialize embedding model
        self.chunking_approach = ingestion_configs["chunking_approach"]
        self.kbs_path = ingestion_configs["kbs_path"]
        self.embed_table = ingestion_configs["embed_table"]
        self.ingest_pip_version = ingestion_configs["ingest_pip_version"]
        # project_root = os.path.abspath(".")
        # if project_root not in sys.path:
        #     sys.path.append(project_root)
        
        print(f"DEBUG: Initialized IngPipeline with kbs_path='{self.kbs_path}', embed_model_id='{self.embed_model_id}', chunk_size={self.chunk_size}, chunking_approach='{self.chunking_approach}'")
        
    # CONFIG METHODS
    def ensure_table_exists(self):
        """
        Creates the document_chunks table if it doesn't exist.
        """
        
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {self.embed_table} (
            id SERIAL PRIMARY KEY,
            document TEXT NOT NULL,
            text TEXT NOT NULL,
            text_hash TEXT GENERATED ALWAYS AS (MD5(text)) STORED,
            pages INTEGER[],
            token_count INTEGER,
            embedding JSONB,
            embeddings_model TEXT,
            kb TEXT,
            chunk_size INTEGER,
            chunking_approach TEXT,
            ingest_pip_version TEXT,
            UNIQUE (text_hash, document, embeddings_model, chunk_size, chunking_approach, ingest_pip_version)
        );
        """
        conn = psql.connect(**self.DB_CONFIG)
        cur = conn.cursor()
        cur.execute(create_table_query)
        conn.commit()
        cur.close()
        conn.close()
        
    def get_embed_model(self):
        """
        Loads and returns the SentenceTransformer model specified by the embed_model_id.

        Returns:
            SentenceTransformer: An instance of the SentenceTransformer model.
        """  
        return SentenceTransformer(self.embed_model_id)    
      
    def get_unique_documents(self):
        """
        Returns a list of unique document configurations from the database.
        Returns tuples of (document, embeddings_model, token_count, chunk_size, chunking_approach, ingest_pip_version)
        """
        try:
            conn = psql.connect(**self.DB_CONFIG)
            cur = conn.cursor()
            cur.execute(f"""
                SELECT DISTINCT document, embeddings_model, token_count, chunk_size, chunking_approach, ingest_pip_version 
                FROM {self.embed_table} 
                GROUP BY document, embeddings_model, token_count, chunk_size, chunking_approach, ingest_pip_version;
            """)
            print("DEBUG: Executed query to fetch unique documents")
            documents = [row for row in cur.fetchall()]
            print(f"DEBUG: Found {len(documents)} unique documents")
            cur.close()
            conn.close()
            return documents
        except Exception as e:
            print(f"Error fetching unique documents: {e}")
            return []  
    
    def delete_document_entries(self, document_title, embeddings_model=None, chunk_size=None, chunking_approach=None, ingest_pip_version=None):
        """
        Deletes entries from the database for a specific document and configuration.
        
        Args:
            document_title (str): Name of the document to delete
            embeddings_model (str, optional): Specific embedding model to match. If None, uses current instance model
            chunk_size (int, optional): Specific chunk size to match. If None, uses current instance chunk size
            chunking_approach (str, optional): Specific chunking approach to match. If None, uses current instance approach
            ingest_pip_version (str, optional): Specific pipeline version to match. If None, uses current instance version
            
        Returns:
            int: Number of entries deleted
        """
        try:
            # Use instance values if parameters not provided
            embeddings_model = embeddings_model or self.embed_model_id
            chunk_size = chunk_size or self.chunk_size
            chunking_approach = chunking_approach or self.chunking_approach
            ingest_pip_version = ingest_pip_version or self.ingest_pip_version
            
            conn = psql.connect(**self.DB_CONFIG)
            cur = conn.cursor()
            
            delete_query = f"""
            DELETE FROM {self.embed_table} 
            WHERE document = %s 
            AND embeddings_model = %s 
            AND chunk_size = %s 
            AND chunking_approach = %s 
            AND ingest_pip_version = %s
            """
            
            cur.execute(delete_query, (document_title, embeddings_model, chunk_size, chunking_approach, ingest_pip_version))
            deleted_count = cur.rowcount
            
            conn.commit()
            cur.close()
            conn.close()
            
            print(f"Deleted {deleted_count} entries for document '{document_title}' with configuration:")
            print(f"  - Embedding Model: {embeddings_model}")
            print(f"  - Chunk Size: {chunk_size}")
            print(f"  - Chunking Approach: {chunking_approach}")
            print(f"  - Pipeline Version: {ingest_pip_version}")
            
            return deleted_count
            
        except Exception as e:
            print(f"Error deleting document entries: {e}")
            return 0
    
    def delete_all_document_entries(self, document_title):
        """
        Deletes ALL entries from the database for a specific document, regardless of configuration.
        
        Args:
            document_title (str): Name of the document to delete all entries for
            
        Returns:
            int: Number of entries deleted
        """
        try:
            conn = psql.connect(**self.DB_CONFIG)
            cur = conn.cursor()
            
            delete_query = f"DELETE FROM {self.embed_table} WHERE document = %s"
            
            cur.execute(delete_query, (document_title,))
            deleted_count = cur.rowcount
            
            conn.commit()
            cur.close()
            conn.close()
            
            print(f"Deleted ALL {deleted_count} entries for document '{document_title}'")
            
            return deleted_count
            
        except Exception as e:
            print(f"Error deleting all document entries: {e}")
            return 0
    
    def drop_and_recreate_table(self):
        """
        Drops the existing table and recreates it with the new structure (including text_hash).
        WARNING: This will delete ALL data in the table!
        """
        try:
            conn = psql.connect(**self.DB_CONFIG)
            cur = conn.cursor()
            
            # Drop the existing table
            drop_query = f"DROP TABLE IF EXISTS {self.embed_table}"
            cur.execute(drop_query)
            print(f"Dropped existing table: {self.embed_table}")
            
            # Create the new table with updated structure
            create_table_query = f"""
            CREATE TABLE {self.embed_table} (
                id SERIAL PRIMARY KEY,
                document TEXT NOT NULL,
                text TEXT NOT NULL,
                text_hash TEXT GENERATED ALWAYS AS (MD5(text)) STORED,
                pages INTEGER[],
                token_count INTEGER,
                embedding JSONB,
                embeddings_model TEXT,
                kb TEXT,
                chunk_size INTEGER,
                chunking_approach TEXT,
                ingest_pip_version TEXT,
                UNIQUE (text_hash, document, embeddings_model, chunk_size, chunking_approach, ingest_pip_version)
            );
            """
            cur.execute(create_table_query)
            print(f"Created new table: {self.embed_table} with text_hash structure")
            
            conn.commit()
            cur.close()
            conn.close()
            
            return True
            
        except Exception as e:
            print(f"Error dropping and recreating table: {e}")
            return False
        
    def process_document(self, document_title,kb):
        """Processes document and returns a DocumentObject with all its pages processed.

        Args:
            document_title (str): document file name

        Returns:
            DocumentObject: Document object
        """
        
        document_path = os.path.join(self.kbs_path,kb,document_title)
        reader = PdfReader(document_path)
        
        page_objects = []
        
        # process each page to generate PageObject
        for i, page in enumerate(reader.pages):
            
            page_text = page.extract_text()
            page_size = int(len(page_text)/4.7)
            page_object = PageObject(page_number=i+1,
                                    text=page_text,
                                    size=page_size,
                                    document=document_title,
                                    kb=kb)
            
            page_objects.append(page_object.model_dump())
        
        # creates DocumentObject with PageObjects' contents
        document_object = DocumentObject(document=document_title,
                                        pages=page_objects,
                                        kb=kb)
        
        return document_object


    def create_chunks(self,document_object,kb):
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
        for i in range(0, total_tokens - self.chunk_size + 1, self.chunk_size):
            chunk_slice = tokens_with_page[i:i+self.chunk_size]
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
                document=document_object.document,
                kb = kb,
                chunk_size=self.chunk_size,
                ingest_pip_version=self.ingest_pip_version,
                chunking_approach=self.chunking_approach,
                embeddings_model=self.embed_model_id
            )
            chunks.append(chunk.model_dump())
            
        return chunks
    
    def create_embeddings(self,chunks,embeddings_model):
        """
        Creates embeddings for each chunk in a list of chunks using a SentenceTransformer model.
        """
        
        for chunk in chunks:
            embedding = embeddings_model.encode(chunk["text"])
            chunk["embedding"] = embedding
            chunk['ingest_pip_version'] = self.ingest_pip_version
            
        return chunks
    
    def save_embeddings(self,embeddings):
        """
        Saves a list of chunks with embeddings into PostgreSQL.
        """
        self.ensure_table_exists()  # Ensure table exists before inserting

        conn = psql.connect(**self.DB_CONFIG)
        cur = conn.cursor()

        insert_query = f"""
        INSERT INTO {self.embed_table} (text, pages, token_count, document, embedding, embeddings_model, kb, chunk_size, chunking_approach, ingest_pip_version)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (text_hash, document, embeddings_model, chunk_size, chunking_approach, ingest_pip_version) DO NOTHING;  -- Skips duplicates based on text hash
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
                        self.embed_model_id,  # Fixed: was self.embeddings_model_id
                        embedding["kb"],
                        embedding["chunk_size"],
                        embedding["chunking_approach"],
                        embedding["ingest_pip_version"]  # Convert list to JSONB
                    ))
                #print(cur.statusmessage)
            except Exception as e:
                print(f"Error inserting embedding: {e}")
                continue

        conn.commit()
        cur.close()
        conn.close()
    
    def ingest_document(self,document_title,kb):
        """
        Ingests a document into the database by processing it, creating chunks, and saving embeddings.
        """
        
        print(f"DEBUG: ingest_document called with document_title='{document_title}', kb='{kb}'")
        print(f"Processing document: {document_title}")
        document_object = self.process_document(document_title,kb)
        #return document_object
        print("Creating chunks...")
        chunks = self.create_chunks(document_object, kb)

        print("loading embeddings model...")
        embeddings_model = self.get_embed_model()
        
        print("Creating embeddings...")
        embeddings = self.create_embeddings(chunks,embeddings_model)
        print("Unloading embeddings model...")
        del embeddings_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
        #return embeddings
        print("Checking table...")
        self.ensure_table_exists()
        
        print("Saving embeddings...")
        self.save_embeddings(embeddings)
        
        print(f"{document_title} has been ingested.")
        print(f"DEBUG: ingest_document completed successfully")
        return embeddings
    
    def ingest_pipeline(self):
        """
        Ingests all documents in the knowledge base folders into the database. 
        If a document has already been ingested with the same configuration, it will be skipped.
        """
        
        # Get existing configurations to avoid duplicates
        unique_configs = self.get_unique_documents()
        print(unique_configs)
        # Process each knowledge base
        if os.path.exists(self.kbs_path):
            print(f"DEBUG: Knowledge base path exists: {self.kbs_path}")
            print("Starting ingestion pipeline...")
            kbs = os.listdir(self.kbs_path)
            
            for kb in kbs:
                kb_path = os.path.join(self.kbs_path, kb)
                print(f"DEBUG: Processing knowledge base: {kb} at path {kb_path}")
                # Skip if not a directory
                if not os.path.isdir(kb_path):
                    print(f"Skipping {kb} as it is not a directory.")
                    continue
                    
                print(f"\nProcessing knowledge base: {kb}")
                kb_documents = os.listdir(kb_path)
                
                for document in kb_documents:
                    # Skip non-PDF files
                    if not document.lower().endswith('.pdf'):
                        continue
                        
                    print(f"Checking document: {document}")
                    
                    # Check if this exact configuration already exists in the database
                    config_exists = False
                    for existing_config in unique_configs:
                        # existing_config format: (document, embeddings_model, token_count, chunk_size, chunking_approach, ingest_pip_version)
                        existing_document = existing_config[0]
                        existing_embed_model = existing_config[1] 
                        existing_chunk_size = existing_config[3]  # chunk_size is at index 3
                        existing_chunking_approach = existing_config[4]
                        existing_pip_version = existing_config[5]
                        
                        # Check if all configuration parameters match
                        if (existing_document == document and 
                            existing_embed_model == self.embed_model_id and
                            existing_chunk_size == self.chunk_size and
                            existing_chunking_approach == self.chunking_approach and
                            existing_pip_version == self.ingest_pip_version):
                            config_exists = True
                            break
                    
                    if not config_exists:
                        print(f"  -> Document {document} needs ingestion.")
                        # Ingest the document
                        try:
                            self.ingest_document(document, kb)
                        except Exception as e:
                            print(f"Error ingesting {document}: {e}")
                    else: 
                        print(f"  -> {document} already ingested with the same configuration.")
        else:
            print(f"Knowledge base path does not exist: {self.kbs_path}")
            

# HELPER CLASSES
##################################################################################
class DocumentObject(BaseModel):
    document: str
    pages: list
    kb: str

class PageObject(BaseModel):
    page_number: int
    text: str
    size: int
    document: str
    kb: str
    
class Chunk(BaseModel):
    text: str
    pages: List[int]
    token_count: int
    document: str
    kb: str
    chunk_size: int
    ingest_pip_version: str
    chunking_approach: str
    embeddings_model: str
##################################################################################