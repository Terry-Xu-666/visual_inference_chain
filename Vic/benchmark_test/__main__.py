import argparse
import yaml
from ..LLM.LLM_state import set_api_client, set_vic_api_client
from ..LLM.LLM_route import API_client
from .benchmark_test import Benchmarktest
from ..benchmark_tool.benchmark import visual_benchmark

parser = argparse.ArgumentParser(description='benchmark_test')

parser.add_argument('--config', default='config.yaml',type=str, help='Model Config')
parser.add_argument('-i','--indicator',default='original', help='Indicator')
parser.add_argument('-p','--benchmark',type=str, help='Benchmark path')
parser.add_argument('-e','--exchange_info_df',default=None,type=str, help='Exchange info df path')

args = parser.parse_args()

# try:
with open(args.config) as file:
    model_config = yaml.load(file, Loader=yaml.FullLoader)
    vic_api_config = model_config.get("vic_api_config",None)
    model_config = model_config["model_config"]
# except:
#     raise ValueError("Model Config is not appropriate set")

set_vic_api_client(API_client(vic_api_config))
set_api_client(API_client(model_config))

args.benchmark = visual_benchmark.from_tsv(args.benchmark)
benchmarktest = Benchmarktest(benchmark=args.benchmark,indicator=args.indicator,exchange_info_df=args.exchange_info_df)
benchmarktest.factory()