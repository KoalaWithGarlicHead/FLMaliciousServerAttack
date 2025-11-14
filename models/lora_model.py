from models import BaseModel
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from peft.tuners.lora import LoraLayer
import torch
import bitsandbytes as bnb
from typing import *
from utils import logger
import torch.nn.init as init
class LoraPeftModel(BaseModel):
    def __init__(self,
                 peft_id: Optional[str]=None,
                 **kwargs):
        super().__init__(**kwargs)

        self.peft_id = peft_id
        self.emb_size = kwargs["r"] if "r" in kwargs.keys() else self.emb_size

        self.model.config.use_cache = False

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'lora_A' in name or 'lora_B' in name:
                    if param.requires_grad:
                        # Xavier initialization
                        init.xavier_normal_(param.data)

    def find_all_linear_names(self, model):
        cls = bnb.nn.Linear4bit if self.bits == 4 else (bnb.nn.Linear8bitLt if self.bits == 8 else torch.nn.Linear)
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if 'lm_head' in lora_module_names:  # needed for 16-bit
            lora_module_names.remove('lm_head')
        return list(lora_module_names)
