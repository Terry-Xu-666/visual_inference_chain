vic_api_client = None
api_client = None
eval_api_client = None

def set_api_client(client):
    global api_client
    api_client = client
    
def get_api_client():
    return api_client

def set_vic_api_client(client):
    global vic_api_client
    vic_api_client = client

def get_vic_api_client():
    # if vic already set,vic_api_client could be None
    return vic_api_client

def set_eval_api_client(client):
    global eval_api_client
    eval_api_client = client

def get_eval_api_client():
    return eval_api_client