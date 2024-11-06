from ..LLM.LLM_state import get_api_client,get_vic_api_client
from pydantic import BaseModel, Field
from .visual_inference_prompt import *
from typing import TypeAlias

base64str : TypeAlias = str

class visual_inference_chain(BaseModel):
    
    inference_instruction: list[str] = Field(
        description="List of instructions for the visual inference chain",
        
    )

class VIC_base():
    
    def __init__(self, query:str, image:list[base64str]|None=None):
        self.api_client = get_api_client()
        self.vic_api_client = get_vic_api_client()
        self.query = query
        self.image = image
        
    def rules_parse(self,response):
    
        try:
            if 'reasoning steps' in response.lower() or 'step-by-step reasoning' in response.lower():
                response = response.split(':')[1:]
                response = ''.join(response)
            response = response.strip().split('\n')
            

            pipeline_list = [i for i in response if not i.isspace() and i != '']
            
            if not pipeline_list:
                pipeline_list = 'error'
            
            
        except:

            pipeline_list = 'error'
        
        return pipeline_list
    
    def rejection_check(self,response):
        
        rejection_keywords = ["cannot provide","cannot answer", "can't provide","can't answer"]
        if any(keyword in response for keyword in rejection_keywords):
            print(f"Received a rejection message. Here is the trace of the responses:\n{response}")
            return universal_prompt
        return response
    
    def visual_chain_merge(self,visual_inference_chain):
        visual_chain_context = visual_analysis_instructions+'reasoning step\n'
        for index,step in enumerate(visual_inference_chain):
                if index == len(visual_inference_chain)-1:
                    visual_chain_last_instruction = last_instruction+f'{step}'
                else:
                    visual_chain_context += f'{index+1}. {step}\n'
        return [visual_chain_context,visual_chain_last_instruction]
    
    
    
    
        
class VIC(VIC_base):
    
    def get_vic(self,retry=1,max_retry=3):
        if retry > max_retry:
            raise ValueError("Max retry limit reached")
        input_data = {'system_prompt': vic_prompt, 'query': self.query}
        response =self.vic_api_client.request(input_data)
        response = self.rejection_check(response)
        visual_inference_pipeline = self.rules_parse(response)
        if visual_inference_pipeline == 'error':
            visual_inference_pipeline = self.get_vic(retry=retry+1)
        return visual_inference_pipeline
    
    def get_vic_with_visual(self):
        input_data = {'system_prompt': vic_prompt_with_visual, 'query': self.query,'image': self.image}
        response =self.vic_api_client.request(input_data)
        response = self.rejection_check(response)
        visual_inference_pipeline = self.rules_parse(response)
        if visual_inference_pipeline == 'error':
            visual_inference_pipeline = self.stucture_decoding(response)
        return visual_inference_pipeline
    
    def only_vic(self):
        visual_inference_pipeline = self.get_vic()
        return {'visual_inference_chain': visual_inference_pipeline}
    
    def only_vic_with_visual(self):
        visual_inference_pipeline = self.get_vic_with_visual()
        return {'visual_inference_chain': visual_inference_pipeline}
    
    def normal_vic(self):
        visual_inference_chain = self.get_vic()
        response = self.vic(visual_inference_chain)
        return response
    
    def visual_all(self):
        visual_inference_chain = self.get_vic_with_visual()
        response = self.vic(visual_inference_chain)
        return response
    
    def vic(self,visual_inference_pipeline):
        visual_chain_context,visual_chain_last_instruction = self.visual_chain_merge(visual_inference_pipeline)
        input_data = {'query': visual_chain_context,'image': self.image}
        intermediate_response = self.api_client.request(input_data)
        final_query = intermediate_instruction.format(question=self.query,extracted_info=intermediate_response,output_format=visual_chain_last_instruction)
        input_data = {'query': final_query,'image': self.image}
        response = self.api_client.request(input_data)
        return {'response': response,'visual_inference_chain': visual_inference_pipeline,'intermediate_response': intermediate_response}
        
    def extract_info(self,extract_info,visual_inference_chain):
        visual_chain_context,visual_chain_last_instruction = self.visual_chain_merge(visual_inference_chain)
        final_query = intermediate_instruction.format(question=self.query,extracted_info=extract_info,output_format=visual_chain_last_instruction)
        input_data = {'query': final_query,'image': self.image}
        response = self.api_client.request(input_data)
        return {'response': response,'visual_inference_chain': visual_inference_chain,'intermediate_response': extract_info}
    
    def vic_m(self,visual_inference_chain):
        intermediate_response = ''
        for index,step in enumerate(visual_inference_chain):
            if index != len(visual_inference_chain)-1:
                instruction = vic_m_context.format(step=step,answers=intermediate_response)
                input_data = {'query': instruction,'image': self.image}
                response = self.api_client.request(input_data)
                intermediate_response += 'step'+str(index+1)+':'+response+'\n'
            else:
                final_query = intermediate_instruction.format(question=self.query,extracted_info=intermediate_response,output_format=step)
                input_data = {'query': final_query,'image': self.image}
                response = self.api_client.request(input_data)
                
        return {'response': response,'visual_inference_chain': visual_inference_chain,'intermediate_response': intermediate_response}
    
    def vic_m_from_scratch(self):
        visual_inference_chain = self.get_vic()
        response = self.vic_m(visual_inference_chain)
        return response
    
    def stucture_decoding(self,response):
        
        prompt = structure_prompt
        input_data = {'system_prompt': prompt, 'query': response,'response_format': visual_inference_chain}
        response =self.vic_api_client.request(input_data)
        return response    
        

class VIC_async(VIC_base):
    async def get_vic(self):
        input_data = {'system_prompt': vic_prompt, 'query': self.query}
        response = await self.vic_api_client(input_data)
        response = self.rejection_check(response)
        visual_inference_pipeline = self.rules_parse(response)
        if visual_inference_pipeline == 'error':
            visual_inference_pipeline = self.structure_decoding(response)
        return visual_inference_pipeline
    
    async def get_vic_with_visual(self):
        input_data = {'system_prompt': vic_prompt_with_visual, 'query': self.query,'image': self.image}
        response = await self.vic_api_client(input_data)
        response = self.rejection_check(response)
        visual_inference_pipeline = self.rules_parse(response)
        if visual_inference_pipeline == 'error':
            visual_inference_pipeline = self.structure_decoding(response)
        return visual_inference_pipeline
    
    async def only_vic(self):
        visual_inference_pipeline = await self.get_vic()
        return {'visual_inference_chain': visual_inference_pipeline}
    
    async def only_vic_with_visual(self):
        visual_inference_pipeline = await self.get_vic_with_visual()
        return {'visual_inference_chain': visual_inference_pipeline}
    
    async def normal_vic(self):
        visual_inference_chain = await self.get_vic()
        response = await self.vic(visual_inference_chain)
        return response
    
    async def visual_all(self):
        visual_inference_chain = await self.get_vic_with_visual()
        response = await self.vic(visual_inference_chain)
        return response
    
    async def vic(self,visual_inference_pipeline):
        visual_chain_context,visual_chain_last_instruction = self.visual_chain_merge(visual_inference_pipeline)
        input_data = {'query': visual_chain_context,'image': self.image}
        intermediate_response = await self.api_client(input_data)
        final_query = intermediate_instruction.format(question=self.query,extracted_info=intermediate_response,output_format=visual_chain_last_instruction)
        input_data = {'query': final_query,'image': self.image}
        response = await self.api_client(input_data)
        return {'response': response,'visual_inference_chain': visual_inference_pipeline,'intermediate_response': intermediate_response}
    
    async def extract_info(self,extract_info,visual_inference_chain):
        visual_chain_context,visual_chain_last_instruction = self.visual_chain_merge(visual_inference_chain)
        final_query = intermediate_instruction.format(question=self.query,extracted_info=extract_info,output_format=visual_chain_last_instruction)
        input_data = {'query': final_query,'image': self.image}
        response = await self.api_client(input_data)
        return {'response': response,'visual_inference_chain': visual_inference_chain,'intermediate_response': extract_info}
    
    async def vic_m(self,visual_inference_chain):
        intermediate_response = ''
        for index,step in enumerate(visual_inference_chain):
            if index != len(visual_inference_chain)-1:
                instruction = vic_m_context.format(step=step,answers=intermediate_response)
                input_data = {'query': instruction,'image': self.image}
                response = await self.api_client(input_data)
                intermediate_response += 'step'+str(index+1)+':'+response+'\n'
            else:
                final_query = intermediate_instruction.format(question=self.query,extracted_info=intermediate_response,output_format=step)
                input_data = {'query': final_query,'image': self.image}
                response = await self.api_client(input_data)
                
        return {'response': response,'visual_inference_chain': visual_inference_chain,'intermediate_response': intermediate_response}
    
    async def vic_m_from_scratch(self):
        visual_inference_chain = await self.get_vic()
        response = await self.vic_m(visual_inference_chain)
        return response
    
    async def stucture_decoding(self,response):
        
        prompt = structure_prompt
        input_data = {'system_prompt': prompt, 'query': response,'response_format': visual_inference_chain}
        response = await self.vic_api_client(input_data)
        return response