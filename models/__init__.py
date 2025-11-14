from models.base_model import BaseModel
from models.lora_model import LoraPeftModel

Model_List = {
    'gpt': LoraPeftModel,
    "llama": LoraPeftModel,
    'gpt2-full': BaseModel,
    "t5-full": BaseModel,
    "bert": BaseModel,
    "hunyuan": BaseModel,
}
Mode_List = {
    "normal": BaseModel,
    "qlora": LoraPeftModel,
    "lora": LoraPeftModel
}

def load_model(config):
    model = Mode_List[config["mode"]](**config)
    return model