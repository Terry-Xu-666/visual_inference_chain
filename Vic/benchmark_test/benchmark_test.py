import os
from tqdm import tqdm
from rich.console import Console
from ..LLM.LLM_state import get_api_client
from ..utils.file import import_file
from ..Vic.Vic_main import Vic, Vic_async
from ..utils.file import to_base64_image

console = Console()

class Benchmarktest():
    
    def __init__(self,**kwargs) -> None:
        
        self.benchmark = kwargs.get("benchmark")
        self.indicator = kwargs.get("indicator")
        self.model = get_api_client()
        self.dirname = self.benchmark.dirname
        self.model_name = self.model.llm.model_name
        self.benchmark_name = self.benchmark.benchmark_name
        self.exchange_info_df = kwargs.get("exchange_info_df",None)
        
    @property
    def result_name(self):
        return os.path.join(self.dirname, f"{self.model_name}_{self.benchmark_name}_{self.indicator}_result.tsv")
    
    def factory(self):
        match self.indicator:
            case "vic":
                self.vic()
            case "vic_ready":
                self.vic_ready()
            case "vic_m":
                self.vic_m()
            case "switch_info":
                self.info_concate()
                self.switch_info()
            case "original":
                self.original()
            case 'cot':
                self.cot()
                
            case _:
                raise ValueError("Indicator not supported")
            
    def checking_exiting_file(self):
        if os.path.exists(self.result_name):
            df = import_file(self.result_name)
            console.print(f"Loaded {len(df)} results from {self.result_name}",style="bold green")
            return len(df)
        else:
            console.print(f"Created {self.result_name}",style="bold green")
            return 0
        
    def original(self):
        existing_len = self.checking_exiting_file()
        benchmark = self.benchmark[existing_len:]
        
        for row in tqdm(benchmark.iterrows(),total=len(benchmark)):
            row_info = row[0]
            row_df = row.left
            prompt = row_info['prompt']
            image = [to_base64_image(row_info['image'])]
            input_data = {'query':prompt,'image':image}
            response = self.model.request(input_data)
            row_df.loc[:,f'{self.model_name}_response'] = response
            self.store(row_df)
    
    def cot(self):
        existing_len = self.checking_exiting_file()
        benchmark = self.benchmark[existing_len:]
        
        for row in tqdm(benchmark.iterrows(),total=len(benchmark)):
            row_info = row[0]
            row_df = row.left
            cot_prompt = 'Answer the following question step by step based on image.' #You should also focus on the information from image. Explain each step of your reasoning process clearly and thoroughly before arriving at the final answer. Even if the problem seems simple, break it down into small, logical steps to ensure clarity and correctness.'
            prompt = cot_prompt+row_info['prompt']
            image = [to_base64_image(row_info['image'])]
            input_data = {'query':prompt,'image':image}
            response = self.model.request(input_data)
            row_df.loc[:,f'{self.model_name}_response'] = response
            self.store(row_df)
        
            
    
    def vic(self):
        existing_len = self.checking_exiting_file()
        benchmark = self.benchmark[existing_len:]
        
        for row in tqdm(benchmark.iterrows(),total=len(benchmark)):
            row_info = row[0]
            row_df = row.left
            prompt = row_info['prompt']
            image = [to_base64_image(row_info['image'])]
            response = Vic(query=prompt,image=image)
            row_df.loc[:,f'visual_chain'] = str(response['visual_inference_chain'])
            row_df.loc[:,f'{self.model_name}_intermediate_response'] = response['intermediate_response']
            row_df.loc[:,f'{self.model_name}_response'] = response['response']
            self.store(row_df)
            
    def vic_ready(self):
        existing_len = self.checking_exiting_file()
        benchmark = self.benchmark[existing_len:]
        
        for row in tqdm(benchmark.iterrows(),total=len(benchmark)):
            row_info = row[0]
            row_df = row.left
            prompt = row_info['prompt']
            image = [to_base64_image(row_info['image'])]
            visual_chain = row.visual_chain.iloc[0]
            response = Vic(query=prompt,image=image,vic=visual_chain)
            row_df.loc[:,f'visual_chain'] = str(response['visual_inference_chain'])
            row_df.loc[:,f'{self.model_name}_intermediate_response'] = response['intermediate_response']
            row_df.loc[:,f'{self.model_name}_response'] = response['response']
            self.store(row_df)
            
    def vic_m(self):
        existing_len = self.checking_exiting_file()
        benchmark = self.benchmark[existing_len:]
        
        for row in tqdm(benchmark.iterrows(),total=len(benchmark)):
            row_info = row[0]
            row_df = row.left
            prompt = row_info['prompt']
            image = [to_base64_image(row_info['image'])]
            visual_chain = row.visual_chain.iloc[0]
            response = Vic(query=prompt,image=image,vic=visual_chain,vic_m=True)
            row_df.loc[:,f'visual_chain'] = str(response['visual_inference_chain'])
            row_df.loc[:,f'{self.model_name}_intermediate_response'] = response['intermediate_response']
            row_df.loc[:,f'{self.model_name}_response'] = response['response']
            self.store(row_df)
            
    def info_concate(self):
        if self.exchange_info_df is None:
            raise ValueError("exchange_info_df is required")
        exchange_info_column = [col for col in self.exchange_info_df.columns if 'intermediate_response' in col]
        self.benchmark.df['exchange_info'] = self.exchange_info_df[exchange_info_column].values.tolist()
        
    def switch_info(self):
        existing_len = self.checking_exiting_file()
        benchmark = self.benchmark[existing_len:]
        
        for row in tqdm(benchmark.iterrows(),total=len(benchmark)):
            row_info = row[0]
            row_df = row.left
            prompt = row_info['prompt']
            image = [to_base64_image(row_info['image'])]
            visual_chain = row.visual_chain.iloc[0]
            exchange_info = row.df.exchange_info.iloc[0]
            response = Vic(query=prompt,image=image,vic=visual_chain,extract_info=exchange_info)
            row_df.loc[:,f'visual_chain'] = str(response['visual_inference_chain'])
            row_df.loc[:,f'{self.model_name}_intermediate_response'] = response['intermediate_response']
            row_df.loc[:,f'{self.model_name}_response'] = response['response']
            self.store(row_df)
            
    def store(self,row_df):
        if not os.path.exists(self.result_name):
            row_df.to_csv(self.result_name,sep='\t',index=False)
        else:
            row_df.to_csv(self.result_name,sep='\t',mode='a',index=False,header=False)
            
        console.print(f"Stored 1 result in {self.result_name}",style="bold green")
        
            
        
        