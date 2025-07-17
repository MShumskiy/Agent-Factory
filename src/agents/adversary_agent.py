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
    
class AdvAgent:
    
    def __init__(self, tools_desc, model, kb_agent):
        
        
        from dotenv import load_dotenv
        from sentence_transformers import CrossEncoder
        
        load_dotenv()
        
        self.src = 'adv_agent'
        self.model = model
        self.llmp_url = os.getenv("LLMP_URL")
        self.llmp_password = os.getenv("LLMP_PASSWORD")
        self.tools_desc = tools_desc
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.kb_agent = kb_agent

        
    def llmp_call(self, prompt, system_prompt, model):
        """ 
        Call the LLMP API to generate a response
        All related to the call is processed here
        """
        
        headers = {
        "Content-Type": "application/json",
        "Authorization": self.llmp_password
    }
        
        # Construct request payload
        request_data = GenerateRequest(
        model=model,
        system_prompt=system_prompt,
        prompt=prompt,
        tools=None,
        src=self.src)
        
        payload = request_data.model_dump(exclude_none=True)

        try:
            response = requests.post(self.llmp_url, headers=headers, json=payload)
            response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None
        
    def adv_agent_response(self, user_prompt,output_format):
        """ 
        Encompasses logic behind tool decision making
        Calls the llmp_call method to generate a response
        """
        
        system_prompt = """You are an adversary agent. You are presented with relevant context and the player's move. Your goal is to counter that move based on the context you are provided."""
        override_config={"system_prompt_rag":system_prompt,
                         'format':output_format}

        
        return self.kb_agent.kb_agent_chat(user_prompt,override_config)
        
        
    def adv_agent_chat(self, user_prompt,output_format=None):
            """
            Logic behind tool activation.
            Sends to agent_0_response for tool decision.
            Activates tool.

            Args:
                user_prompt (str)
            """

            user_prompt = user_prompt.strip('Need an adversary')
            # user_prompt = "Counter this" + user_prompt + """\n Provide the response in the following JSON format:
            # {Player:<player_name>,
            # 'moves':{<move name>:<very direct move description>}
            # }"""
            response,references,output = self.adv_agent_response(user_prompt,output_format) # for testing
            #response = self.adv_agent_response(user_prompt)
            return response,references,output # for testing
            return response
                
                
        