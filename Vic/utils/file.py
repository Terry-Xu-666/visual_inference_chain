import re
import os
import requests
import base64
import io
import pandas as pd
import json
from PIL import Image
import tempfile


def to_base64_image(image):
    if isinstance(image, str):
        if re.match(r'http[s]?://', image):
            return base64.b64encode(requests.get(image).content).decode('utf-8')
        elif os.path.exists(image):
            with open(image, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        else:
            return base64.b64encode(base64.b64decode(image)).decode('utf-8')
    elif isinstance(image, io.BytesIO):
        return base64.b64encode(image.getvalue()).decode('utf-8')
    else:
        raise ValueError('image type error')
    
def import_file(path):
    if path.endswith('.tsv'):
        return pd.read_csv(path,sep='\t')
    elif path.endswith('.json'):
        return pd.read_json(path)
    elif path.endswith('.csv'):
        return pd.read_csv(path)
    elif path.endswith('.jsonl'):
        with open(path, 'r') as f:
            lines = f.readlines()
        data = [json.loads(line) for line in lines]
        return pd.DataFrame(data)
    else:
        raise ValueError('File format not supported')
    
def binary_to_image(binary):
    return io.BytesIO(base64.b64decode(binary))

def temp_path(Byte_IO):
    image=Image.open(Byte_IO).convert('RGB')
    temp_file=tempfile.NamedTemporaryFile(delete=False,suffix='.jpg')
    image.save(temp_file.name,'JPEG')
    return temp_file.name