from .base_client import Client

CLIENTS = {
    "base": Client,
}

def load_client(client_idx, dataset, client_config):
    return CLIENTS[client_config["name"].lower()](client_idx, dataset, **client_config)