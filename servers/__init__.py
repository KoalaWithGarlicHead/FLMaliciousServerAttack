from servers.base_server import Server

SERVERS = {
    "base": Server,
}

def load_server(server_config):
    return SERVERS[server_config["name"].lower()](**server_config)