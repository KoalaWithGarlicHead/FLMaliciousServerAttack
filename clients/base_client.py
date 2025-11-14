from typing import *

class Client(object):
    def __init__(
            self,
            client_idx,
            dataset,
            client_config: Optional[dict] = {"name": "base"},
            **kwargs
    ):
        self.client_idx = client_idx
        self.client_config = client_config
        self.dataset = dataset

    def get_data(self):
        return self.dataset


