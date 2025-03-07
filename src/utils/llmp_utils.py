# ASGENT 0 SIDE NOT RAG
from pydantic import BaseModel
import requests
import os
from typing import Optional, List, Dict

llmp_url = os.getenv("LLMP_URL")
llmp_password = os.getenv("LLMP_PASSWORD")


class GenerateRequest(BaseModel):
    model: str
    system_prompt: str = ''
    prompt: str
    format: Optional[dict] = None
    image: Optional[str] = None
    tools: Optional[List[Dict]] = None
    src: str = None
    temperature: float = 0.5

def llmp_call(prompt, system_prompt, model,temperature=0.5):
        """ 
        Call the LLMP API to generate a response
        All related to the call is processed here
        """
        
        headers = {
        "Content-Type": "application/json",
        "Authorization": llmp_password
    }
        
        # Construct request payload
        request_data = GenerateRequest(
        model=model,
        system_prompt=system_prompt,
        prompt=prompt,
        tools=None,
        src="RAG test",
        temperature=temperature)
        
        payload = request_data.model_dump(exclude_none=True)

        try:
            response = requests.post(llmp_url, headers=headers, json=payload)
            response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None