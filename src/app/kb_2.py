import os
import json
import requests
import gradio as gr
import shutil
import sys
from fpdf import FPDF
from PyPDF2 import PdfFileMerger
from unidecode import unidecode 

project_root = os.path.abspath(".")
print(project_root)
if project_root not in sys.path:
    sys.path.append(project_root)
#from src.agents.agent_0 import Agent0
from src.pipelines.ingestion_pipeline import ingest_pipeline
from src.agents.kb_agent_standalone import KBAgent



# def save_uploaded_document(pdf_file):
#     """
#     Saves an uploaded PDF file to the data/documents directory.
    
#     Args:
#         pdf_file (dict): A dictionary containing the uploaded file information 
#                          (e.g., with keys 'name' and 'data' where 'data' is the file path).
                         
#     Returns:
#         str: The full destination path where the file was saved.
#     """
#     # Determine the repository root assuming this file is in src/app/
#     root_dir = os.path.abspath(project_root)
#     docs_path = os.path.join(root_dir, "data", "documents")
    
#     # Create the documents directory if it doesn't exist
#     os.makedirs(docs_path, exist_ok=True)
#     print(docs_path)
#     # Extract file name and source path from the uploaded file object
#     filename = os.path.basename(pdf_file)
#     destination_path = os.path.join(docs_path, filename)
    
#     # Define destination file path
#     destination_path = os.path.join(docs_path, filename)
    
#     # Copy the file to the destination
#     shutil.copy(pdf_file, destination_path)
    
#     return destination_path

knowledge_bases_desc = {
    'physics_kb':'a knowledge base with information related to physics',
    'mathematics_kb':'a knowledge base with information related to mathematics',
    'economics_kb':'a knowledge base with information related to economics and business',
    'military_kb':'a knowledge base with information related to military, war and strategy',
    'ai_kb':'a knowledge base with information related to artificial intelligence and machine learning',
    'psychology_kb':'a knowledge base with information related to psychology and human behavior',
    'law_kb':'a knowledge base with information related to law and legal systems',
    }

model = 'gemma3:4b'

kb_agent = KBAgent(knowledge_bases_desc,model)

with gr.Blocks() as demo:
    gr.Markdown("""## Knowledge Base Agent
                """)
    status_box = gr.Markdown("")

    with gr.Row():

        with gr.Column(scale=1):  # Buttons on the left
            job_input = gr.Textbox(
                label="Your Request",
                lines=1,
                elem_id="styled-input"
                )
            response_output = gr.Markdown(elem_id="styled-output")
            references_output = gr.Markdown(label="References",elem_id="references-output")
    with gr.Row():
        #with gr.Column(scale=1):  # Output and download on the right
            # kb_selector = gr.CheckboxGroup(
            #     label="Knowledge Base Selection",
            #     choices=["physics_kb", "mathematics_kb", "economics_kb",
            #              "military_kb","ai_kb"],  # Dummy entries
            #     value=["physics_kb"],
            #     interactive=True,
            #     elem_id="kb-selector"
            #     )
        trigger_ingestion_btn = gr.Button(
                    "Trigger Ingestion",
                    elem_id="process-documents"
                    )
            # threshold_input = gr.Textbox(
            #         label="Threshold",
            #         placeholder="In progress",
            #         lines=1
            #         )
        generate_btn = gr.Button("Run",
                                     elem_id="generate-button")
            # download_button = gr.DownloadButton(
            #     label="⬇ Download PDF (not working yet)",
            #     elem_id="custom-download")
        


    # Style the download button
    gr.HTML("""
            <style>
            #styled-output {
                background-color: #2E2A24;
                border: 1px solid #ccc;
                padding: 12px;
                border-radius: 4px;
                font-family: monospace;
                white-space: pre-wrap;
                height: 550px;
                overflow-y: auto;
            }
            #styled-input textarea {
                background-color: #2E2A24;
                border: 1px solid #ccc;
                padding: 4px;
                border-radius: 4px;
                font-family: monospace;
                white-space: pre-wrap;
                height: 100px;
                overflow-y: auto;
            }

            #custom-download button {
                background-color: #E5BA6F;
                color: white;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: bold;
                border: none;
                margin-top: 10px;
            }
            #references-output {
                background-color: #1E1C19;
                border: 1px solid #aaa;
                padding: 12px;
                border-radius: 4px;
                font-family: monospace;
                white-space: pre-wrap;
                height: 120px;
                overflow-y: auto;
                font-size: 5px
            }
            #references-output * {
                # font-size: 5px !important;
                # }
            #kb-selector .wrap {
                flex-direction: column !important;
                align-items: flex-start;
            }

            #custom-download button:hover {
                background-color: #E5BA6F;
            }
            </style>
            """)
    def run_ingestion():
        status_box_text = "🔄 Processing documents... Please wait."
        yield status_box_text
        ingest_pipeline()
        yield "✅ Documents successfully processed!"
        
    # Trigger
    #generate_btn = gr.Button("Run")
    job_input.submit(kb_agent.kb_agent_chat, [job_input], [response_output, references_output])
    generate_btn.click(kb_agent.kb_agent_chat, [job_input], [response_output,references_output])
    trigger_ingestion_btn.click(fn=run_ingestion, inputs=None, outputs=status_box)  # Run with no inputs/outputs


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
