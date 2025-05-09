from fastapi import FastAPI, HTTPException, Header, Depends, Request
from pydantic import BaseModel
from dotenv import load_dotenv
import os

import requests
from typing import List, Dict, Optional
from src.utils.llmp_utils import llmp_call

class GenerateRequest(BaseModel):
    model: str
    system_prompt: str = ''
    prompt: str
    format: Optional[dict] = None
    image: Optional[str] = None
    tools: Optional[List[Dict]] = None
    src: str = None
    temperature: float = 0.5
    
class KBAgent:
    
    def __init__(self, tools_desc, model):
        
        
        from dotenv import load_dotenv
        from sentence_transformers import CrossEncoder
        
        load_dotenv()
        
        
        self.src = 'kb_agent'
        self.model = model
        self.llmp_url = os.getenv("LLMP_URL")
        self.llmp_password = os.getenv("LLMP_PASSWORD")
        self.tools_desc = tools_desc
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        
    def kb_agent_response(self, user_prompt, temperature=0):
        """ 
        Encompasses logic behind tool decision making
        Calls the llmp_call method to generate a response
        """
        
        system_prompt = "You are a selector agent- Your task is to select the most appropriate knowledge base to answer the query. You provide a very short and direct answer, no unnecessary text."
        
        tools_description = "\n ".join([f"{key}: {value}" for key, value in self.tools_desc.items()])
        prompt = f"{user_prompt}\nwhich of the following knowledge bases would you use?\n {tools_description}. Provide only the name of the selected knowledge base and nothing else."
        
        llmp_response = llmp_call(prompt, system_prompt, 'llama3.2:latest', temperature,src = 'KB Agent')['message']['content']
        
        results = self.cross_encoder.predict([[llmp_response, tool] for tool in self.tools_desc.keys()])
        
        tool_scores = dict(zip(self.tools_desc.keys(), results))
        best_tool = max(tool_scores, key=tool_scores.get)
        
        
        return best_tool
        
        
    def kb_agent_chat(self, user_prompt,override_config=None):
            """
            Logic behind tool activation.
            Sends to agent_0_response for tool decision.
            Activates tool.

            Args:
                user_prompt (str)
            """
            print('KB Agent')
            selected_kb = self.kb_agent_response(user_prompt)

            
                
            from src.pipelines.rag_pipeline import generate_rag
                
            user_prompt_rag = user_prompt.strip('given my documents')
            print('RAG pipeline')

            rag_output = generate_rag(self.model, user_prompt_rag,selected_kb,override_config)  
                #return rag_output,user_prompt,selected_kb
            llmp_response = rag_output[0]['message']['content']
                # if llmp_response is not None:
                #     break
                # else:
                #     print(f"Attempt {attempt + 1} failed. Retrying...")
                #     if attempt == max_retries - 1:
                #         raise Exception("Max retries exceeded. Unable to get a valid response.")

                
            references = rag_output[1]
            references_output = ""
            for doc, pages in references.items():
                references_output += f"📄 **{doc}**\n"
                references_output += f"   📑 Pages: {', '.join(map(str, pages))}\n\n"  
            # print(llmp_response)
                
            # print("📚 References:\n")
            # for doc, pages in references.items():
            #     print(f"📄 **{doc}**")
            #     print(f"   📑 Pages: {', '.join(map(str, pages))}\n")
            
            output = "\n".join([llmp_response,f"📚 References:\n{references_output}"])
            return llmp_response,references,output # testing
            #return output # could make sense to return decoupled response & refs for frontend
                
                
        