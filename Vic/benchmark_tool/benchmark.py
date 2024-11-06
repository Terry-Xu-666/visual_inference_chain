import os
import pandas as pd
from tqdm import tqdm
import ast
import pandas as pd
from rich.console import Console
from rich import print
from rich.pretty import Pretty
from tqdm import tqdm

from ..utils.file import import_file
from ..LLM.LLM_state import get_vic_api_client
from ..Vic.Vic_main import Vic

console = Console()


class visual_benchmark():
    
    def __init__(self,data:pd.DataFrame,path:str|None = None) -> None:
        '''
        Load the benchmark file.The benchmark file should be a tsv or json file with the following columns or keys:
        prompt: The prompt for the question
        image: The image for the question in base64 format
        other columns: Any other columns that are required for the benchmark
        '''
        if path is not None:
            self.path = path
            self.dirname = os.path.dirname(path)
            self.benchmark_name = os.path.basename(path).split('.')[0]
            
        self.df = data
        self.vic_model = get_vic_api_client()
        

        if 'visual_chain' in self.df.columns:
            self._visual_chain_path = path
            self._visual_chain = self.visual_chain_list()
        else:
            self._visual_chain_path = None
            self._visual_chain = None
        
        if 'inference_chain_with_visual' in self.df.columns:
            self._visual_chain_with_visual_path = path
            self._visual_chain_with_visual = self.visual_chain_with_visual()
        else:
            self._visual_chain_with_visual = None
            self._visual_chain_with_visual_path = None
        
    
    @property
    def prompt(self):
            
        return self.df['prompt']
    


    @property
    def left(self):
        mask = ~self.df.columns.isin(['image'])
    
        return self.df.loc[:,mask]
    
    @property
    def visual_chain(self):
        if self._visual_chain is None:
            raise ValueError("There has been no visual chain provided, please generate a visual chain first")
        elif len(self._visual_chain) != len(self):
            raise ValueError("There are lack of visual chains for some of the prompts")
        return self._visual_chain
    
    
   
    
    @property
    def visual_chain_with_visual_path(self):
        if 'inference_chain_with_visual' in os.path.basename(self.path) and self.vic_model.llm.model_name in os.path.basename(self.path):
            self._visual_chain_with_visual_path = self.path
        elif self._visual_chain_with_visual_path is None:
            self._visual_chain_with_visual_path = os.path.join(self.dirname,self.benchmark_name+self.vic_model.llm.model_name+"_inference_chain_with_visual.tsv")
        return self._visual_chain_with_visual_path
    @property
    def visual_chain_path(self):
        if f"{self.vic_model.llm.model_name}_visual_chain" in os.path.basename(self.path):
            self._visual_chain_path = self.path
        elif self._visual_chain_path is None:
            self._visual_chain_path = os.path.join(self.dirname,self.benchmark_name+self.vic_model.llm.model_name+"_visual_chain.tsv")
        return self._visual_chain_path
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,index):
        
        if isinstance(index,slice):
            new_instance = visual_benchmark(self.df.iloc[index].reset_index(drop=True),self.path)
            return new_instance
        
        if isinstance(index,int):
            return {'prompt':self.prompt[index],'image':self.df.iloc[index,self.df.columns.get_loc('image')]}   
    
    def iterrows(self,batch_size=1):
        length = len(self)
        for i in range(0,length,batch_size):
                yield self[i:i+batch_size]
                
                
    def visual_chain_list(self):
        column_name = 'visual_chain'
        if isinstance(self.df[column_name].iloc[0],list):
            return self.df[column_name]
        else:
            self.df[column_name] = self.df[column_name].apply(lambda x:ast.literal_eval(x) if pd.notna(x) else x)
            return self.df[column_name]
        
    def visual_chain_with_visual(self):
        if isinstance(self.df['inference_chain_with_visual'].iloc[0],list):
            return self.df['inference_chain_with_visual']
        else:
            self.df['inference_chain_with_visual'] = self.df['inference_chain_with_visual'].apply(lambda x:ast.literal_eval(x) if pd.notna(x) else x)
            return self.df['inference_chain_with_visual']
    
    def checking_existing_file(self):
        if self.visual_chain_path == self.path:
            return None
        if os.path.exists(self.visual_chain_path):
            console.print("Visual chain file already exists",style="bold blue")
            if console.input('Do you want to start with this existing file? (y/n)') == 'y':
                self.df = import_file(self.visual_chain_path)
                self.path = self.visual_chain_path
                if 'visual_chain' in self.df.columns:
                    self._visual_chain = self.visual_chain_list()
                if 'inference_chain_with_visual' in self.df.columns:
                    self._visual_chain_with_visual = self.visual_chain_with_visual()
                      
    def add_visual_chain(self):
        self.checking_existing_file()
        try:

            if self.vic_model is None:
                console.print("Please set the api_client first",style="bold red")
                return None
            if self._visual_chain is not None:
                console.print("Visual chain already exists",style="bold red")
                if self.visual_chain_path == self.path:
                    if pd.isna(self._visual_chain).any():
                        console.print("There are lack of visual chains for some of the prompts",style="bold red")
                    else:
                        return None
                elif console.input('The existing visual chains are not matching with the model you used before. However, they are incomplete. Do you want to overwrite them by current setting vic chain model? (y/n)') == 'n':
                    return None
            
            console.print("Adding visual chain to the benchmark",style="bold blue")
            for i, row in tqdm(self.df.iterrows(),total=len(self)):
                visual_chain = self.generate_visual_chain(row)
                self.df.loc[i,'visual_chain'] = visual_chain
            self.df.to_csv(self.visual_chain_path,sep='\t',index=False)
            
        except Exception as e:
            console.print("Error occured while adding visual chain",style="bold red")
            console.print(e,style="bold red")
            self.df.to_csv(self.visual_chain_path,sep='\t',index=False)
            console.print('Saved the current state of the benchmark',style="bold blue")
        except KeyboardInterrupt:
            console.print("Process interrupted",style="bold red")
            self.df.to_csv(self.visual_chain_path,sep='\t',index=False)
            console.print('Saved the current state of the benchmark',style="bold blue")
            
    def checking_existing_visual_file(self):
        if self.visual_chain_with_visual_path == self.path:
            return None
        if os.path.exists(self.visual_chain_with_visual_path):
            console.print("Visual chain with visual file already exists",style="bold blue")
            if console.input('Do you want to start with this existing file? (y/n)') == 'y':
                self.df = import_file(self.visual_chain_with_visual_path)
                self._visual_chain_with_visual_path = self.path
                if 'inference_chain_with_visual' in self.df.columns:
                    self._visual_chain_with_visual = self.visual_chain_with_visual()
                    
    def add_visual_chain_with_visual(self):
        self.checking_existing_visual_file()
        try:
            if self.vic_model is None:
                console.print("Please set the api_client first",style="bold red")
                return None
            if self._visual_chain_with_visual is not None:
                console.print("Visual chain already exists",style="bold red")
                if 'inference_chain_with_visual' in self.df.columns:
                    if self._visual_chain_with_visual_path == self.path:
                        console.print("There are lack of visual chains for some of the prompts",style="bold red")
                elif console.input('The existing visual chains are not matching with the model you used before. However, they are incomplete. Do you want to overwrite them by current setting vic chain model? (y/n)') == 'n':
                    return None
                
            console.print("Adding visual chain to the benchmark",style="bold blue")
            for i, row in tqdm(self.df.iterrows(),total=len(self)):
                visual_chain = self.generate_visual_chain_with_visual(row)
                print(Pretty(visual_chain))
                self.df.loc[i,'inference_chain_with_visual'] = visual_chain
            self.df.to_csv(self.visual_chain_with_visual_path,sep='\t',index=False)
        
        except Exception as e:
            console.print("Error occured while adding visual chain",style="bold red")
            console.print(e,style="bold red")
            self.df.to_csv(self.visual_chain_with_visual_path,sep='\t',index=False)
            console.print('Saved the current state of the benchmark',style="bold blue")
        except KeyboardInterrupt:
            console.print("Process interrupted",style="bold red")
            self.df.to_csv(self.visual_chain_with_visual_path,sep='\t',index=False)
            console.print('Saved the current state of the benchmark',style="bold blue")
            
    def generate_visual_chain(self,row):
        column_name = 'visual_chain'
        if column_name in row.index:
           if row[column_name]:
               try:
                    if not pd.isna(row[column_name]).any():
                        return str(row[column_name])
               except:
                   pass
            
        
        prompt = row['prompt']
        visual_chain = Vic(prompt,only_vic=True)['visual_inference_chain']
        print(Pretty(visual_chain))
        return str(visual_chain)
            
        
    def generate_visual_chain_with_visual(self,row):
        column_name = 'inference_chain_with_visual'
        if column_name in row.index:
           if row[column_name]:
               try:
                    if not pd.isna(row[column_name]).any():
                        return str(row[column_name])
               except:
                   pass
        
        prompt = row['prompt']
        image = [row['image']]
        visual_chain = Vic(prompt,image,only_vic_with_visual=True)['visual_inference_chain']
        return str(visual_chain)


            
    @classmethod
    def from_tsv(cls,path):
        data = import_file(path)
        return cls(data,path)