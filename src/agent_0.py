from fastapi import FastAPI, HTTPException, Header, Depends, Request
from pydantic import BaseModel
from dotenv import load_dotenv
import os

import requests
from typing import List, Dict, Optional
from src.utils.llmp_utils import llmp_call
from src.agents.kb_agent import KBAgent
from src.agents.adversary_agent import AdvAgent

class GenerateRequest(BaseModel):
    model: str
    system_prompt: str = ''
    prompt: str
    format: Optional[dict] = None
    image: Optional[str] = None
    tools: Optional[List[Dict]] = None
    src: str = None
    temperature: float = 0.5
    
class Agent0:
    
    def __init__(self, tools_desc, model, agents):
        
        
        from dotenv import load_dotenv
        from sentence_transformers import CrossEncoder
        
        load_dotenv()
        
        
        self.src = 'agent_0'
        self.model = model
        self.llmp_url = os.getenv("LLMP_URL")
        self.llmp_password = os.getenv("LLMP_PASSWORD")
        self.tools_desc = tools_desc
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        self.knowledge_bases_desc = {'physics_kb':'a knowledge base with information related to physics',
              'mathematics_kb':'a knowledge base with information related to mathematics',
              'economics_kb':'a knowledge base with information related to economics and business',
              'military_kb':'a knowledge base with information related to military, war and strategy',
              }
        print("Initializing Agents!")
        for agent in agents:
            if agent == 'kb_agent':
                self.kb_agent = KBAgent(self.knowledge_bases_desc,self.model)
            if agent == 'adv_agent':
                self.adv_agent = AdvAgent(self.knowledge_bases_desc,self.model,self.kb_agent)
        print("Agents are ready for your use!")
        
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
        
    def agent_0_response(self, user_prompt, temperature=0):
        """ 
        Encompasses logic behind tool decision making
        Calls the llmp_call method to generate a response
        """
        
        system_prompt = "You are an assistant. Your only goal is to provide me the name of the tool I would need to get the answer to the prompt and nothing else."
        
        tools_description = "\n ".join([f"{key}: {value}" for key, value in self.tools_desc.items()])
        prompt = f"{user_prompt}\nwhich of the following tools would you use?\n {tools_description}"
        
        llmp_response = llmp_call(prompt,
                                  system_prompt,
                                  'llama3.2:latest',
                                  temperature,
                                  src = 'Agent 0')['message']['content']
        
        results = self.cross_encoder.predict([[llmp_response, tool] for tool in self.tools_desc.keys()])
        
        tool_scores = dict(zip(self.tools_desc.keys(), results))
        best_tool = max(tool_scores, key=tool_scores.get)
        
        
        return best_tool,user_prompt
        
        
    def agent_0_chat(self, user_prompt):
            """
            Logic behind tool activation.
            Sends to agent_0_response for tool decision.
            Activates tool.

            Args:
                user_prompt (str)
            """
            
            agent_0_response = self.agent_0_response(user_prompt)
            
            selected_tool = agent_0_response[0]
            print(f"Passing to: \n{selected_tool} !")
            
            if selected_tool == 'image_generator':
                
                from src.image_generator import ImageGeneratorAgent
                img_gen = ImageGeneratorAgent()
                img_gen.generate(user_prompt)
                
            if selected_tool == 'ingestion_pipeline':
                
                from src.pipelines.ingestion_pipeline import ingest_pipeline
                ingest_pipeline()
            
            if selected_tool == 'Knowledge Base Query Agent':                
                
                #user_prompt_rag = user_prompt.strip('given my documents')
                return self.kb_agent.kb_agent_chat(user_prompt)
            if selected_tool == 'Adversary Agent':
                
                return self.adv_agent.adv_agent_chat(user_prompt)
                
                
                
                
        