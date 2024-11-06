from typing import TypedDict,List,Any, TypeAlias,Dict
from typing_extensions import NotRequired
import google.generativeai as genai
import os
from PIL import Image
import requests
from io import BytesIO
import base64
from openai import OpenAI, AzureOpenAI, AsyncOpenAI, AsyncAzureOpenAI
import anthropic
import os
from pydantic import BaseModel
import json
import ast
import filetype


from ..logging.error import error_handler,error_handler_async


    
LLM_output : TypeAlias = str
ModelConfig : TypeAlias = Dict[str, Any]
json_schema : TypeAlias = BaseModel

class LLM_input(TypedDict):
    system_prompt: NotRequired[str]
    image: NotRequired[List[Any]]
    query : str
    response_format: NotRequired[json_schema]
    
class LLM():
    def __init__(self,
                 model_name : str,
                 api_keys : str|None,
                 Config: ModelConfig|None=None) -> None:
        
        self.model_name = model_name
        self.api_keys = api_keys
        self.Config = Config
        
    @error_handler
    def predict(self, input: LLM_input) -> LLM_output:
        
        response = self.API_server(input)
        
        return response
    
    @error_handler_async
    async def predict_async(self, input: LLM_input) -> LLM_output:
        
        response = await self.API_server_async(input)
        
        return response
    
class OPENAI(LLM):
    
    def __init__(self, 
                 model_name: str, 
                 api_keys: str | None,
                 Config: ModelConfig|None=None) -> None:
        
        super().__init__(model_name, api_keys)
        if api_keys is None:
            api_keys = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_keys)
        self.client_async = AsyncOpenAI(api_key=api_keys)
    
    @error_handler
    def API_server(self, input: LLM_input) -> LLM_output:
            
        messages = self.messages(input)
        
        if input.get('response_format'):
            response = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                response_format=input.get('response_format')
            )
            
            return response.choices[0].message.parsed.model_dump_json()
        else:
            response = self.client.chat.completions.create(
                model = self.model_name,
                messages =messages
                )
        
            return response.choices[0].message.content.strip()
    
    @error_handler_async
    async def API_server_async(self, input: LLM_input) -> LLM_output:
        
        messages = self.messages(input)
        
        if input.get('response_format'):
            response = await self.client_async.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                response_format=input.get('response_format')
            )
            return response.choices[0].message.parsed.model_dump_json()
        else:
        
            response = await self.client_async.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
        
            return response.choices[0].message.content.strip()
    
    def messages(self, input: LLM_input) -> List[dict]:
        
        messages = []
        if input.get("system_prompt"):
            messages.append({
                "role": "system",
                "content": input["system_prompt"]
            })
        content =[{"type": "text","text": input["query"]}]
        
        if input.get('image'): #image should be a list of base64 encoded images
            image_file = [image for image in input.get('image')]
            for image in image_file:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image}"
                            }
                        }
                    )
        messages.append({"role": "user","content": content})
        
        return messages
    
class AZURE(OPENAI):
    def __init__(self, 
                 api_keys: str | None,
                 Config: ModelConfig|None) -> None:
        if api_keys is None:
            api_keys = os.getenv("AZURE_API_KEY")
        self.api_keys = api_keys
        self.Config = Config
        self.model_name = self.Config.get('deployment_name')
        self.api_base = self.Config.get('api_base')
        self.api_version = self.Config.get('api_version')
        self.organization = self.Config.get('organization')
        self.client = AzureOpenAI(
                    api_key=self.api_keys,  
                    api_version=self.api_version,
                    base_url=f"{self.api_base}/openai/deployments/{self.model_name}"
                )
        self.client_async = AsyncAzureOpenAI(
                    api_key=self.api_keys,  
                    api_version=self.api_version,
                    base_url=f"{self.api_base}/openai/deployments/{self.model_name}"
                )
        
        
class Gemini(LLM):
    
    def __init__(self, 
                 model_name: str, 
                 api_keys: str | None,
                 Config: ModelConfig|None) -> None:
        
        super().__init__(model_name, api_keys)
        if api_keys is None:
            GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')
        genai.configure(api_key=GOOGLE_API_KEY)
        self.client = genai.GenerativeModel(model_name)
    
    @error_handler
    def API_server(self, input: LLM_input) -> LLM_output:
        
        messages = self.messages(input)
        
        response = self.client.generate_content(messages)
        
        return response.text
    
    @error_handler_async
    async def API_server_async(self, input: LLM_input) -> LLM_output:
        
        messages = self.messages(input)
        
        response = await self.client.generate_content_async(messages)
        
        return response.text
    
    def messages(self, input: LLM_input) -> List[Any]:
        
        query = ''
        if input.get("system_prompt"):
            query = 'system_prompt:\n'+input["system_prompt"]
        query = query + '\nquery:\n'+input["query"]
        content = [query]
        if input.get('image'):
            for image in input['image']:
                image = self.binary_to_image(image)
                content.append(image)
        return content
    
    def binary_to_image(self, binary: str) -> Image:
        return Image.open(BytesIO(base64.b64decode(binary)))
    
class Qwen(OPENAI):
    
    def __init__(self, 
                 model_name: str, 
                 api_keys: str | None,
                 Config: ModelConfig|None) -> None:
        self.model_name = model_name
        self.api_keys = api_keys
        if self.api_keys is None:
            self.api_keys = os.getenv("DASHSCOPE_API_KEY")
            
        self.client = OpenAI(
            api_key=self.api_keys,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.client_async = AsyncOpenAI(
            api_key=self.api_keys,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    @error_handler
    def API_server(self, input: LLM_input) -> LLM_output:
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages(input)
        )
        
        return response.choices[0].message.content.strip()
    
    @error_handler_async
    async def API_server_async(self, input: LLM_input) -> LLM_output:
        
        response = await self.client_async.chat.completions.create(
            model=self.model_name,
            messages=self.messages(input)
        )
        
        return response.choices[0].message.content.strip()
    
class Claude(LLM):
    
    def __init__(self, 
                 model_name: str, 
                 api_keys: str | None,
                 Config: ModelConfig|None) -> None:
        
        super().__init__(model_name, api_keys)
        if api_keys is None:
            api_keys = os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=api_keys)
        
    @error_handler
    def API_server(self, input: LLM_input) -> LLM_output:
        
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=2048,
            messages=self.messages(input)
        )
        
        return response.content[0].text
    
    def messages(self, input: LLM_input) -> List[dict]:
        messages = []
        if input.get("system_prompt"):
            messages.append({
                "role": "system",
                "content": input["system_prompt"]
            })
        content=[]
        if input.get("image"):
            images = [image for image in input.get("image")]
            for image in images:
                media_type = filetype.guess(base64.b64decode(image)).mime
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        'media_type': media_type,
                        "data": image,
                    },
                })
        content.append({
            "type": "text",
            "text": input["query"]
        })
        messages.append({
            "role": "user",
            "content": content
        })
        return messages
    

    
    
    
    
    
    

    
