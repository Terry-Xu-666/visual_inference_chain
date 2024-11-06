from typing import Any
import asyncio
import time

from ..logging.error import error_handler_async,error_handler
from .LLM import LLM, ModelConfig, OPENAI, AZURE, Gemini, Qwen, LLM_input, Together_ai,Local_model,LLM_output,Claude

class LLM_route():
    
    def __call__(self,
                 config : ModelConfig) -> LLM:
        if config is None:
            return None
        service_provider = config.get("service_provider")
        model_name = config.get("model_name")
        api_keys = config.get("api_keys",None)
        
        match service_provider:
            case "openai":
                return OPENAI(model_name, api_keys, config)
            case "azure":
                return AZURE(api_keys, config)
            case "gemini":
                return Gemini(model_name, api_keys, config)
            case "qwen":
                return Qwen(model_name, api_keys, config)
            case 'claude':
                return Claude(model_name, api_keys, config)
            case _:
                raise ValueError("Service provider not supported")
            
class API_client():
    
    def __init__(self, 
                 config : ModelConfig) -> None:
        
        if config:
            self.llm = LLM_route()
            self.llm = self.llm(config)
            self.rate_limit = config.get("rate_limit",50)
            self.semaphore = asyncio.Semaphore(self.rate_limit)
            self.last_reset = time.time()
        
            
    @error_handler_async
    async def __call__(self, input: LLM_input) -> LLM_output:
        
        async with self.semaphore:
            response = await self.llm.predict_async(input)
            
        return response
    
    @error_handler
    def request(self, input: LLM_input) -> LLM_output:
        
        response = self.llm.predict(input)
        
        return response
    
