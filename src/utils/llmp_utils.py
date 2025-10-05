# ASGENT 0 SIDE NOT RAG
from pydantic import BaseModel
import requests
import os
from typing import Optional, List, Dict
from requests.exceptions import RequestException, HTTPError
import time

llmp_url = os.getenv("LLMP_URL")
llmp_password = os.getenv("LLMP_PASSWORD")
print(llmp_url)

class GenerateRequest(BaseModel):
    model: str
    system_prompt: str = ''
    prompt: str
    format: Optional[dict] = None
    image: Optional[str] = None
    tools: Optional[List[Dict]] = None
    src: str = None
    temperature: float = 0.5
    max_gen_length: int = -1

def llmp_call(prompt, system_prompt, model,temperature=0.5,src=None,format=None,max_gen_length = -1):
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
        src=src,
        temperature=temperature,
        format = format,
        max_gen_length = max_gen_length)

        payload = request_data.model_dump(exclude_none=True)

        max_retries = 3
        for attempt in range(max_retries):
            print(llmp_url)
            try:
                response = requests.post(llmp_url, headers=headers, json=payload)
                response.raise_for_status()  # raises HTTPError for bad HTTP responses (e.g., 500)
                return response.json()
            except HTTPError as http_err:
                print(f"❌ HTTP error: {http_err}")
                print(f"📦 Response content:\n{response.text}")

            except RequestException as req_err:
                print(f"⚠️ Request exception: {req_err}")

            except Exception as e:
                print(f"💥 Unexpected error: {e}")

            # Wait and retry
            print(f"🔁 Waiting 60 seconds before retry...")
            time.sleep(60)

        raise Exception("🚨 Max retries exceeded. Unable to get a valid response.")
                
        # try:
        #     response = requests.post(llmp_url, headers=headers, json=payload)
        #     response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
        #     return response.json()
        # except requests.exceptions.RequestException as e:
        #     print(f"Request failed: {e}")
        #     raise e
        
def llmp_list_call():
        """ 
        Call the LLMP API to get list of models
        """
        
        base_url = "http://192.168.1.69:8000"  # Change to your API's base URL
        endpoint = "/models"
        url = base_url + endpoint

        # If your API requires a token for authentication, set it here.
        # Adjust the header key and token as needed.
        headers = {
            "Authorization": llmp_password
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            models = response.json()
            print("Available models:", models)
        else:
            print("Error:", response.status_code, response.text)