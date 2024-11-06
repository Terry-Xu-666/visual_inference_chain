from openai import OpenAI
import os
from tqdm import tqdm
import pandas as pd
import re
import ast
import pandas as pd
import regex
from Levenshtein import distance
from copy import deepcopy
from collections import defaultdict
from retry import retry
from rich.console import Console
import langfun as lf 
import pyglove as pg
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix,f1_score

from ..utils.file import import_file
from ..LLM.LLM_state import get_eval_api_client


console = Console()

class benchmark_eval():
    
    def __init__(self,
                 data : pd.DataFrame,
                 path : str) -> None:
        
        self.df = data
        self.path = path
        self.response = self.find_response()
        self.answer = self.find_answer()
        self.response_answer_pair = list(zip(self.response,self.answer))
        self.model = get_eval_api_client()
        
        self.type = None
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,index):
        return self.response_answer_pair[index]
    @property
    def basename(self):
        return '.'.join(os.path.basename(self.path).split('.')[0:-1])
    
    @property
    def dirname(self):
        return os.path.dirname(self.path)+'/eval_results'
    @property
    def result_path(self):
        basename = self.basename +'_eval.tsv'
        results_path = os.path.join(self.dirname,basename)
        return results_path
    
    @property
    def score_name(self):
        score_df = self.basename + '_eval.xlsx'
        results_path = os.path.join(self.dirname,score_df)
        return results_path
        
    
    def cut(self,index):
        return self.__class__(self.df.iloc[index:].reset_index(drop=True),self.path)
    
    def itterows(self):
        length = len(self)
        for i in range(0,length):
            yield i,self[i]
            
    def get(self,index):
        return self.df.iloc[[index],:]
    
    def get_prompt(self,index):
        return self.df['prompt'][index]
            
    def find_response(self):
        pass
    
    def find_answer(self):
        pass
    
    def find_choice(self,
                    under_eval_bench:'benchmark_eval',
                    index:int) -> str:
        pass
    
    def score(self):
        pass
    
    def eval(self):
        self.Eval_main()
        self.score()
    
    def checking_exiting_file(self):
        if os.path.exists(self.result_path):
            df = import_file(self.result_path)
            console.print(f"Loaded {len(df)} results from {self.result_path}",style="bold green")
            return len(df)
        else:
            console.print(f"Created {self.result_path}",style="bold green")
            return 0
    
    def Eval_main(self):
        # type of evaluation: YORN, MCQ, etc
        
        existing_length = self.checking_exiting_file()
        if existing_length == len(self):
            console.print(f'All response evaluations have been saved. No further evaluations needed.',style='bold green')
        
        else:
            
            eval_benchmark = self.cut(existing_length)
            
            for index,row in tqdm(eval_benchmark.itterows(),total=len(eval_benchmark)):
                response,answer = row
                
                if self.type == 'YORN':
                    LLM_parse,sign = self.YORN(response,answer)
                elif self.type == 'MCQ':
                    LLM_parse,sign = self.MCQ(response,answer,eval_benchmark,index)
                elif self.type == 'MIX':
                    LLM_parse,sign = self.MIX(response,answer,eval_benchmark,index)
                elif self.type == 'Free_Form':
                    LLM_parse,sign = self.Free_Form(response,answer,eval_benchmark,index)
                elif self.type == 'VQA':
                    LLM_parse,sign = self.VQA(response,answer)
                else:
                    raise ValueError('Invalid evaluation type')
                
                answer_df= deepcopy(eval_benchmark.get(index))
                answer_df.loc[:,'LLM_parse'] = LLM_parse
                answer_df.loc[:,'sign'] = sign
                
                self.saved(answer_df)
                
            results = import_file(self.result_path)
            assert len(results) == len(self.df)
            print(f'All response evaluations saved.')
            
    def saved(self,answer_df):
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)
        if not os.path.exists(self.result_path):
            answer_df.to_csv(self.result_path,sep='\t')
        else:
            answer_df.to_csv(self.result_path,sep='\t',mode='a',header=False)
        console.print(f"Stored 1 result in {self.result_path}",style="bold green")
        
        
        
    
    def YORN(self,response,answer):
        
        prompt = "You will be provided with an response to a yes/no question. Your task is to interpret the response and output either 'yes' or 'no,' matching the meaning of the response. If the response shows high confidence in the answer, output 'yes.' If the response shows some confidence in the answer, output 'yes.' If the response shows low confidence in answer, like 'likely,' 'probably,' 'maybe,' 'possibly,' etc., output 'yes.' If the response shows high confidence in the answer, output 'no.' If the response shows some confidence in the answer, output 'no.' If the response shows some confidence in the answer, like 'likely,' 'probably,' conclude in corresponding yes or no. only if the response is not present or very ambiguous, output 'unclear'."
        
        response = f"Response:{response}"
        
        LLM_parse = self.LLM(response,prompt)
        
        sign = self.extract_answer(LLM_parse,answer)
            
        return LLM_parse,sign
    
    
    def MCQ(self,response,answer,under_eval_bench,index):
        
        prompt = "Please read the user's response to a multiple-choice question. Your task is to identify and output the letter (A, B, C, D, etc.) that corresponds to the option chosen by the user. If the response does not clearly indicate a choice or no option is mentioned, output 'unclear'. Only output a single letter or 'unclear'."
        
        choice = self.find_choice(under_eval_bench,index)
        
        response = f"Choises:{choice}\nResponse:{response}"
        # response = f"Response:{response}"
        
        LLM_parse = self.LLM(response,prompt)
        # print(LLM_parse)
        
        sign = self.extract_answer(LLM_parse,answer)
            
        return LLM_parse,sign
    
    def number(self,response,answer):
        
        prompt = """You will be provided with an response to a question that requires a numerical answer. It can be a int, float or a list. Your task is to indentify and parse the final numerical answer. You answer should only contain the answer of number(or number list) with no punctuations. Make sure your answer is in the correct format and is as accurate as possible. if you can not find the number or it is unclear, output 'unclear'
        Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.
        Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
        Question: Which number is missing?

        Model response: The number missing in the sequence is 14.

        Extracted answer: 14

        Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.
        Question: What is the fraction of females facing the camera?

        Model response: The fraction of females facing the camera is 0.6, which means that six out of ten females in the group are facing the camera.        

        Extracted answer: 0.6

        Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.
        Question: How much money does Luca need to buy a sour apple candy and a butterscotch candy? (Unit: $)

        Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.

        Extracted answer: 1.45

        Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.      
        Question: Between which two years does the line  graph saw its maximum peak?

        Model response: The line graph saw its maximum peak between 2007 and 2008.

        Extracted answer: [2007, 2008]
        """
        response = 'response:'+response+'Extracted answer:'
        
        
        LLM_parse = self.LLM(response,prompt)
        
        
        sign = self.extract_answer(LLM_parse,answer,num=True)
        
        return LLM_parse,sign
    
    def extract_answer(self,LLM_parse,answer,num=False):
        
        if num:
            try:
                answer = int(answer)
                num_type = 'int'
            except ValueError:
                try:
                    answer = float(answer)
                    num_type = 'float'
                except ValueError:
                    try:
                        answer = ast.literal_eval(answer)
                        num_type = 'list'
                    except:
                        num_type = 'str'
                    
            if num_type == 'int':
                try:
                    LLM_parse = int(LLM_parse)
                    if LLM_parse == answer:
                        return 1
                    else:
                        return 0
                except ValueError:
                    return 0
                
            elif num_type == 'float':
                try:
                    LLM_parse = float(LLM_parse)
                    if LLM_parse == answer:
                        return 1
                    else:
                        return 0
                except ValueError:
                    return 0
                
            elif num_type == 'list':
                try:
                    LLM_parse = ast.literal_eval(LLM_parse)
                    if LLM_parse == answer:
                        return 1
                    else:
                        return 0
                except:
                    return 0
            else:
                return 0
            
        else:
            
            LLM_parse = LLM_parse.lower()
            answer = answer.lower()
            
            char='.()[],:;!*#{}'
            for c in char:
                LLM_parse=LLM_parse.replace(c,' ')
                
            LLM_parse = re.sub(r'\s+', ' ', LLM_parse).strip()
                
            LLM_parse_list = LLM_parse.split(' ')
            
            
            if answer in LLM_parse_list:
                
                return 1
            
            else:
                return 0
    
    def VQA(self,response,answer):
            
            prompt = """You will be provided a real answer and a user's response to a visual question. Your task is to determine whether the user's response is correct or incorrect. If the user's response is correct, output 'correct'. If the user's response is incorrect, output 'incorrect'. If the user's response is unclear or ambiguous, output 'unclear'."""
            
            response = f"Real Answer:{answer},Response:{response}"
            
            def is_response_unclear(response):
                # define the maximum number of times a word can be repeated in the response
                max_word_repetition = 10
                
                # split the response into words
                words = response.split(' ')
                
                # count the number of times each word appears in the response
                word_count = {}
                for word in words:
                    if word in word_count:
                        word_count[word] += 1
                    else:
                        word_count[word] = 1
                
                # check if any word is repeated more than the maximum number of times
                for count in word_count.values():
                    if count > max_word_repetition:
                        return "unclear"
                
                # check if the response contains any words that are not alphanumeric
                return response
            
            if is_response_unclear(response) == 'unclear':
                LLM_parse = 'unclear'
            else:
                LLM_parse = self.LLM(response,prompt)
            
            
            
            print(LLM_parse)
            
            sign = 1 if LLM_parse == 'correct' else 0
                
            return LLM_parse,sign
    
    
    @retry(tries=3,delay=2)
    def LLM(self,response,prompt=None):
        if prompt:
        
            data = {'query':response,'system_prompt':prompt}
        
        else:
            data = {'query':response}
        
        output = self.model.request(data)
        
        return output
        
    @classmethod
    def from_tsv(cls,path):
        data = import_file(path)
        return cls(data,path)
    
class Hallucination_eval(benchmark_eval):
    
    def __init__(self,data,path):
        super().__init__(data,path)
        # Initialize the type of evaluation, support for 'YORN' and 'MCQ' right now.
        self.type = 'YORN'
        
    def find_response(self):
        # you must rewrite this function to find the response in your data and return it as a list
        response_column = [col for col in self.df.columns if 'response' in col and 'intermediate' not in col]
        return self.df[response_column].squeeze().tolist()
    
    def find_answer(self):
        # you must rewrite this function to find the answer in your data and return it as a list
        answer_list = self.df['gt_answer'].tolist()
        answer = ['yes' if ans == 1 else 'no' for ans in answer_list]
        self.df['answer'] = answer
        return answer
    
    def score(self):
        # you must define how to score the evaluation results. This is a example for hallucination benchmark.
        try:
            df = import_file(self.result_path)
        except:
            raise ValueError('No evaluation results found.')
        # calculate the related score
        score_dict = {}
        # correct is the number of correct responses
        score_dict['correct'] = df['sign'].sum()
        # total is the total number of responses
        score_dict['total'] = len(df)
        # accuracy is the ratio of correct responses to total responses
        score_dict['accuracy'] = score_dict['correct']/score_dict['total']
        # # if all the answer of a fig is correct, then return is correct (an another statistics)
        # fig_accuracy = df.groupby(['category', "subcategory", "set_id", "figure_id"]).apply(lambda x: 1 if x['sign'].sum()==len(x) else 0)
        # fig_accuracy = fig_accuracy.sum()/len(fig_accuracy)
        # score_dict['fig_accu'] = fig_accuracy
        # save the score to a excel file
        
        #fine-grained evaluation
        category = df.groupby('subcategory').apply(lambda x: x['sign'].sum()/len(x))
        category = category.to_dict()
        score_dict.update(category)
        y_true , y_pred = self.label_parse(df)
        precision = precision_score(y_true,y_pred)
        recall = recall_score(y_true,y_pred)
        f1 = f1_score(y_true,y_pred)
        yes_rate = y_pred.count(1)/len(y_pred)
        score_dict['precision'] = precision
        score_dict['recall'] = recall
        score_dict['f1'] = f1
        score_dict['yes_rate'] = yes_rate
        
        
        score_df = pd.DataFrame([score_dict])
        score_df.to_excel(self.score_name)
        
        #print the score
        for key,value in score_dict.items():
            print(f'{key}:{value}')
        
    def label_parse(self,df):
        y_pred = df['LLM_parse'].tolist()
        y_true = df['gt_answer'].tolist()
        y_pred = [1 if 'yes' in pred.lower() else 0 for pred in y_pred]
        return y_true,y_pred
        
        
    
    
        


class MME_eval(benchmark_eval):
    
    def __init__(self,data,path):
        super().__init__(data,path)
        # Initialize the type of evaluation, support for 'YORN' and 'MCQ' right now.
        self.type = 'YORN'
        
    def find_response(self):
        # you must rewrite this function to find the response in your data and return it as a list
        response_column = [col for col in self.df.columns if 'response' in col and 'intermediate' not in col]
        return self.df[response_column].squeeze().tolist()
    
    def find_answer(self):
        # you must rewrite this function to find the answer in your data and return it as a list
        # answer_list = self.df['ground_truth'].tolist()
        # answer = ['yes' if ans == 1 else 'no' for ans in answer_list]
        # self.df['answer'] = answer
        return self.df['ground_truth'].tolist()
    
    def score(self):
        eval_type_dict = {
            "Perception": ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"],
            "Cognition": ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]
        }

        eval_df = import_file(self.result_path)
        MME_score = dict()

        # you must define how to score the evaluation results. This is a example for hallucination benchmark.
        for eval_type, task_name_list in eval_type_dict.items():
            print("===========", eval_type, "===========")
           
            scores = 0
            task_score_dict = dict()

            for task_name in task_name_list:
                task_df = eval_df[eval_df['subset'] == task_name]

                task_other_ans_num = 0
                task_score = 0
                acc_plus_correct_num = 0
                gts = []
                preds = []

                # response_column = [col for col in self.task_df.columns if 'response' in col and 'intermediate' not in col]
                gts = task_df['ground_truth'].squeeze().tolist()
                gts = list(map(lambda x: x.lower(), gts))
                preds = task_df['LLM_parse'].tolist()
                preds = list(map(lambda x: x.lower(), preds))

                metric_dict = self.compute_metric(gts, preds)
                acc_plus = self.get_accplus(task_df) / len(task_df)
                metric_dict["acc_plus"] = acc_plus
                
                for k, v in metric_dict.items():
                    if k in ["acc", "acc_plus"]:
                        task_score += v*100
                
                task_score_dict[task_name] = task_score
                
                scores += task_score

            print("total score:", scores, "\n")
            for task_name, score in task_score_dict.items():
                print("\t", task_name, " score:", score)
            print("\n")
            
            MME_score[f"{eval_type}_totol"] = scores
            MME_score.update(task_score_dict)
            
        MME_score_df = pd.DataFrame([MME_score])
        MME_score_df.to_excel(self.score_name, index=False)
            
    
    
    def compute_metric(self, gts, preds):
        assert len(gts) == len(preds)

        label_map = {
            "yes": 1,
            "no": 0,
            "unclear": -1,
        }
        
        gts = [label_map[x] for x in gts]
        preds = [label_map[x.strip('.')] for x in preds]

        acc = accuracy_score(gts, preds) 

        clean_gts = []
        clean_preds = []
        other_num = 0 
        for gt, pred in zip(gts, preds):
            if pred == -1:
                other_num += 1
                continue
            clean_gts.append(gt)
            clean_preds.append(pred)
        

        conf_mat = confusion_matrix(clean_gts, clean_preds, labels=[1,0])
        precision = precision_score(clean_gts, clean_preds, average='binary')
        recall = recall_score(clean_gts, clean_preds, average='binary')
        tp, fn = conf_mat[0]
        fp, tn = conf_mat[1]

        metric_dict = dict()
        metric_dict = {
            "TP": tp,
            "FN": fn,
            "TN": tn,
            "FP": fp,
            "precision": precision,
            "recall": recall,
            "other_num": other_num,
            "acc": acc,
        }

        return metric_dict
    
    def get_accplus(self, task_df):
        temp_df = task_df.groupby('picture_name')['sign'].sum()

        acc_plus = (temp_df == 2).sum()

        return acc_plus


            
class Pope_eval(benchmark_eval):
    
    def __init__(self,data,path):
        super().__init__(data,path)
        self.type = 'YORN'
        
    def find_response(self):
        response_column = [col for col in self.df.columns if 'response' in col and 'intermediate' not in col]
        return self.df[response_column].squeeze().tolist()
    
    def find_answer(self):
        answer_list = self.df['ground_truth'].tolist()
        answer_list = [x.lower().strip() for x in answer_list]
        return answer_list
    
    def score(self):
        # Load the evaluation results
        eval_results = import_file(self.result_path)
        
        # Initialize the score dictionaries
        pope_score = {'total':len(eval_results)}
        pope_score['correct'] = eval_results['sign'].sum()
        pope_score['accuracy'] = pope_score['correct']/pope_score['total']
        
        #fine-grained evaluation
        category = eval_results.groupby('subset').apply(lambda x: x['sign'].sum()/len(x))
        category = category.to_dict()
        pope_score.update(category)
        
        y_true, y_pred = self.lable_parse(eval_results)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        yes_rate = y_pred.count(1)/len(y_pred)
        pope_score['f1'] = f1
        pope_score['precision'] = precision
        pope_score['recall'] = recall
        pope_score['yes_rate'] = yes_rate
        
        
        # Print and save the scores
        pope_score_df = pd.DataFrame([pope_score])
        pope_score_df.to_excel(self.score_name, index=False)
        
        for key, value in pope_score.items():
            print(f'{key}: {value:.4f}'+ '\n')
        
    def lable_parse(self,eval_results):
        y_true = eval_results['ground_truth'].tolist()
        y_pred = eval_results['LLM_parse'].tolist()
        y_true = [1 if 'yes' in x.lower() else 0 for x in y_true]
        y_pred = [1 if 'yes' in x.lower() else 0 for x in y_pred]
        return y_true, y_pred
        

class MMbench_eval(benchmark_eval):
    
    def __init__(self,data,path):
        super().__init__(data,path)
        self.type = 'MCQ'
        
    def find_response(self):
        response_column = [col for col in self.df.columns if 'response' in col and 'intermediate' not in col]
        return self.df[response_column].squeeze().tolist()
    
    def find_answer(self):
        answer_list = self.df['answer'].tolist()
        answer_list = [f'({ans.lower()})' for ans in answer_list]
        return answer_list
    
    def find_choice(self,index):
        return self.df.loc[index,'choices']
    
    def MCQ(self,response,answer,under_eval_bench,index):
        
        prompt = "Please read the user's response to a multiple-choice question. Your task is to identify and output the lower letter with parentheses (a), (b),(c),(d) etc. that corresponds to the option chosen by the response. If the response does not clearly indicate a choice or no option is mentioned, output 'unclear'. Only output a single letter or 'unclear'."
        
        choice = under_eval_bench.find_choice(index)
        
        response = f"Choises:{choice}\nResponse:{response}"
        # response = f"Response:{response}"
        
        LLM_parse = self.LLM(response,prompt)
        # print(LLM_parse)
        
        sign = self.extract_answer(LLM_parse,answer)
            
        return LLM_parse,sign
    
    def extract_answer(self, LLM_parse, answer):
        if answer in LLM_parse.lower():
            return 1
        else:
            return 0
        
    def score(self):
        
        # Load the evaluation results
        eval_results = import_file(self.result_path)
        
        # Initialize the score dictionaries
        MMbench_score = {'total':len(eval_results)}
        MMbench_score['correct'] = eval_results['sign'].sum()
        MMbench_score['accuracy'] = MMbench_score['correct']/MMbench_score['total']
        
        #fine-grained evaluation
        category = eval_results.groupby('category').apply(lambda x: x['sign'].sum()/len(x))
        category = category.to_dict()
        MMbench_score.update(category)
        
        l2_category = eval_results.groupby('l2-category').apply(lambda x: x['sign'].sum()/len(x))
        l2_category = l2_category.to_dict()
        MMbench_score.update(l2_category)
        
        # Print and save the scores
        MMbench_score_df = pd.DataFrame([MMbench_score])
        MMbench_score_df.to_excel(self.score_name, index=False)
        
        for key, value in MMbench_score.items():
            print(f'{key}: {value:.4f}'+ '\n')
            
class Seed_bench_eval(benchmark_eval):
    
    def __init__(self,data,path):
        super().__init__(data,path)
        self.type = 'MCQ'
        
    def find_response(self):
        response_column = [col for col in self.df.columns if 'response' in col and 'intermediate' not in col]
        return self.df[response_column].squeeze().tolist()
    
    def find_answer(self):
        answer_list = self.df['answer'].tolist()
        answer_list = [f'({ans.lower()})' for ans in answer_list]
        return answer_list
    
    def find_choice(self,index):
        return self.df.loc[index,'choices']
    
    def MCQ(self,response,answer,under_eval_bench,index):
        
        prompt = "Please read the user's response to a multiple-choice question. Your task is to identify and output the upper letter with parentheses (A), (B),(C),(D) etc. that corresponds to the option chosen by the response. If the response does not clearly indicate a choice or no option is mentioned, output 'unclear'. Only output a single letter or 'unclear'."
        
        choice = under_eval_bench.find_choice(index)
        
        response = f"Choises:{choice}\nResponse:{response}"
        # response = f"Response:{response}"
        
        LLM_parse = self.LLM(response,prompt)
        # print(LLM_parse)
        
        sign = self.extract_answer(LLM_parse,answer)
            
        return LLM_parse,sign
    
    def extract_answer(self, LLM_parse, answer):
        if answer in LLM_parse.lower():
            return 1
        else:
            return 0
    
    def score(self):
            
        # Load the evaluation
        eval_results = import_file(self.result_path)
        
        
        # Initialize the score dictionaries
        seed_bench_score = {'total':len(eval_results)}
        seed_bench_score['correct'] = eval_results['sign'].sum()
        seed_bench_score['accuracy'] = seed_bench_score['correct']/seed_bench_score['total']
        
        #fine-grained evaluation
        category = eval_results.groupby('question_type').apply(lambda x: x['sign'].sum()/len(x))
        category = category.to_dict()
        seed_bench_score.update(category)
        
        # Print and save the scores
        seed_bench_score_df = pd.DataFrame([seed_bench_score])
        seed_bench_score_df.to_excel(self.score_name, index=False)
        
        for key, value in seed_bench_score.items():
            print(f'{key}: {value:.4f}'+ '\n')
            
    
        
        
        
        
        

                
        