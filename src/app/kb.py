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
#from src.agent_factory.agents.agent_0 import Agent0
from src.agent_factory.pipelines.ingestion_pipeline import ingest_pipeline
from src.agent_factory.agents.kb_agent import KBAgent
from src.agent_factory.agents.adversary_agent import AdvAgent
from src.agent_factory.agents.sim_agent import SIMAgent



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

agent_0_tools_desc = {
    'Simulation Agent':'a simulation agent that simulates a game between two players. Triggered by command "Simulate a scenario.". Pay strict attention to the explicit command! If the command is not in the request then its not this tool!',
    'Adversary Agent':'an adversary agent that is used to act as an adversary to the users strategies. Triggered by command "Need an adversary". Pay strict attention to the explicit command! If the command is not in the request then its not this tool!',
    'image_generator':'a tool that generates images based on a given prompt. Should be triggered by explicit calls like "generate me an image of". Pay strict attention to the explicit command! If the command is not in the request then its not this tool!',
    # 'ingestion_pipeline':'a tool that ingests documents, triggered by command "trigger ingestion". Pay strict attention to the explicit command! If the command is not in the request then its not this tool!',
    'Knowledge Base Query Agent':'a tool that generates answers based on documents, triggeres by command "given my documents,". Pay strict attention to the explicit command! If the command is not in the request then its not this tool!'
              }

knowledge_bases_desc = {'physics_kb':'a knowledge base with information related to physics',
              'mathematics_kb':'a knowledge base with information related to mathematics',
              'economics_kb':'a knowledge base with information related to economics and business',
              'military_kb':'a knowledge base with information related to military, war and strategy',
              }

model = 'gemma3:4b'

kb_agent = KBAgent(knowledge_bases_desc,model)

with gr.Blocks() as demo:
    gr.Markdown("""## Knowledge Base Agent\n
                < instructions to use >
                """)
    status_box = gr.Markdown("")

    with gr.Row():
        with gr.Column(scale=1):  # Buttons on the left
            with gr.Row():  # ✅ Side-by-side Trigger Button and Iterations Input
                trigger_ingestion_btn = gr.Button(
                    "Trigger Ingestion",
                    elem_id="process-documents"
                    )
                threshold_input = gr.Textbox(
                    label="Threshold",
                    placeholder="In progress",
                    lines=1
                    )
                rand_e_chance_input = gr.Textbox(
                    label="Placeholder",
                    placeholder="In progress",
                    lines=1
                    )
            job_input = gr.Textbox(
                label="Your Request",
                lines=29,
                elem_id="styled-input"
                )
            generate_btn = gr.Button("Run",
                                     elem_id="generate-button")
        with gr.Column(scale=1):  # Output and download on the right
            response_output = gr.Markdown(elem_id="styled-output")
            references_output = gr.Markdown(elem_id="references-output")
            download_button = gr.DownloadButton(
                label="⬇ Download PDF (not working yet)",
                elem_id="custom-download")


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
                height: 705px;
                overflow-y: auto;
            }
            #styled-input textarea {
                background-color: #2E2A24;
                border: 1px solid #ccc;
                padding: 4px;
                border-radius: 4px;
                font-family: monospace;
                white-space: pre-wrap;
                height: 550px;
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
                height: 300px;
                overflow-y: auto;
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
    generate_btn.click(kb_agent.kb_agent_chat, [job_input], [response_output,references_output])
    trigger_ingestion_btn.click(fn=run_ingestion, inputs=None, outputs=status_box)  # Run with no inputs/outputs


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
