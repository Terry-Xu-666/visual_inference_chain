import argparse
import os
from rich import print
import yaml

from .benchmark_eval import *
from ..LLM.LLM_state import set_eval_api_client
from ..LLM.LLM_route import API_client
parser = argparse.ArgumentParser(description='Evaluate hallucination benchmark')
parser.add_argument('-p','--path', type=str, help='Path to the result file')
parser.add_argument('-b','--bench', type=str, help='the specific benchmark to evaluate')
parser.add_argument('--config', default='config.yaml',type=str, help='Model Config')
args = parser.parse_args()

try:
    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        eval_api_config = config["eval_api_config"]
except:
    raise ValueError("Eval Model Config is not appropriate set")

set_eval_api_client(API_client(eval_api_config))

if os.path.isdir(args.path):
    file_list =[os.path.join(args.path,file) for file in os.listdir(args.path) if file.endswith('.tsv')]
else:
    file_list = [args.path]

print(f'[bold green]There are {len(file_list)} files to evaluate, they are {file_list}')

for file in file_list:
    match args.bench:
        case 'hallubench':
            evaluator = Hallucination_eval.from_tsv(file)
        case 'mathvista':
            evaluator = mathvista_eval.from_tsv(file)
        case 'MME':
            evaluator = MME_eval.from_tsv(file)
        case 'mmvp':
            evaluator = MMVP_eval.from_tsv(file)
        case 'pope':
            evaluator = Pope_eval.from_tsv(file)
        case 'seed':
            evaluator = Seed_bench_eval.from_tsv(file)
    # evaluator.score()
    evaluator.eval()
