import argparse
import yaml
from .LLM.LLM_route import API_client
from .LLM.LLM_state import set_api_client, set_vic_api_client
from .Vic.Vic_main import Vic
from .utils.file import to_base64_image



parser = argparse.ArgumentParser(description='VIC')

parser.add_argument('-q','--query', type=str, help='Query')
parser.add_argument('-i','--image', default=None,type=str, help='Image')
parser.add_argument('--config', default='config.yaml',type=str, help='Model Config')
parser.add_argument('--vic_m', action='store_true', help='VIC_M')
parser.add_argument('--only_vic', action='store_true', help='Only VIC')
parser.add_argument('--only_vic_with_visual', action='store_true', help='Only VIC with visual')
parser.add_argument('--visual_all', action='store_true', help='Visual All')

args = parser.parse_args()

try:
    with open(args.config) as file:
        model_config = yaml.load(file, Loader=yaml.FullLoader)
        vic_api_config = model_config.get("vic_api_config",None)
        model_config = model_config["model_config"]
except:
    raise ValueError("Model Config is not appropriate set")


set_vic_api_client(API_client(vic_api_config))
set_api_client(API_client(model_config))



if args.image is not None:
    args.image = to_base64_image(args.image)
    
    
response = Vic(query=args.query,
                image=[args.image],
                vic_m=args.vic_m,
                only_vic=args.only_vic,
                only_vic_with_visual=args.only_vic_with_visual,
                visual_all=args.visual_all)


if args.only_vic or args.only_vic_with_visual:
    print(response['visual_inference_chain'])
else:
    print(response['response'])