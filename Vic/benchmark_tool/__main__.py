import argparse
import yaml
from ..LLM.LLM_route import API_client
from ..LLM.LLM_state import set_api_client, set_vic_api_client,get_vic_api_client
from .benchmark import visual_benchmark

parser = argparse.ArgumentParser(description='add visual chain to benchmark')
parser.add_argument('--config', default='config.yaml',type=str, help='Model Config')
parser.add_argument('-p','--benchmark',type=str, help='Benchmark path')
args = parser.parse_args()
try:
    with open(args.config) as file:
        model_config = yaml.load(file, Loader=yaml.FullLoader)
        vic_api_config = model_config["vic_api_config"]
except:
    raise ValueError("Model Config is not appropriate set")

set_vic_api_client(API_client(vic_api_config))



visual_benchmark = visual_benchmark.from_tsv(args.benchmark)

visual_benchmark.add_visual_chain()