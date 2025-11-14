
import torch
import os
from typing import *
from abc import abstractmethod


class Trainer(object):

    def __init__(
        self, 
        name: Optional[str] = "Base",
        model_type: Optional[str] = "gpt",
        model_name: Optional[str] = "gpt2",
        lr: Optional[float] = 10e-5,
        weight_decay: Optional[float] = 0.,
        epochs: Optional[int] = 10,
        batch_size: Optional[int] = 4,
        gradient_accumulation_steps: Optional[int] = 1,
        max_grad_norm: Optional[float] = 1.0,
        warm_up_epochs: Optional[int] = 3,
        client_dataset: Optional[str] = "sst-2",
        server_dataset: Optional[str] = "sst-2",
        max_sequence_length: Optional[int] = 1024,
        max_steps: Optional[int] = -1,
        save_steps: Optional[int] = 200,
        logging_steps: Optional[int] = 50,
        save_path: Optional[str] = "../results/",
        train_type: Optional[str] = "server",
        save_path_suffix: Optional[str] = "",
        save_total_limit: Optional[int] = 2,
        lr_scheduler_type: Optional[str]="cosine",
        save_strategy: Optional[str]='steps',
        training_mode: Optional[str]='normal',
        lora_r: Optional[int]=None,
        **kwargs,
    ):

        self.name = name
        self.model_type = model_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.warm_up_epochs = warm_up_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_sequence_length = max_sequence_length
        self.trainer = None
        self.data_collator = None
        self.max_steps = max_steps
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.save_total_limit = save_total_limit
        self.lr_scheduler_type = lr_scheduler_type
        self.save_strategy = save_strategy

        self.data_collator_name = "DataCollatorForCausalLM"
        if "data_collator_name" in kwargs.keys():
            self.data_collator_name = kwargs["data_collator_name"]

        if save_path_suffix != "": save_path_suffix = "-"+save_path_suffix
        if training_mode in ["normal"]:
            self.total_save_path = f'{save_path}{name}-{model_type}-{model_name}-{training_mode}-{client_dataset}-{server_dataset}{save_path_suffix}'
        elif training_mode in ["lora", "qlora"] and lora_r is not None:
            self.total_save_path = f'{save_path}{name}-{model_type}-{model_name}-{training_mode}-r={lora_r}-{client_dataset}-{server_dataset}{save_path_suffix}'
        os.makedirs(self.total_save_path, exist_ok=True)
        self.save_path = f'{self.total_save_path}/{train_type}'
        os.makedirs(self.save_path, exist_ok=True)
        self.loss_log_path = f'{self.total_save_path}/{train_type}/loss_log.txt'
        self.diff_save_dir = f'{self.total_save_path}/{train_type}/diff_save_dir'
        os.makedirs(self.diff_save_dir, exist_ok=True)

        # path
        if training_mode in ["normal"]:
            self.adapter_save_path = os.path.join(self.save_path, "full_model")
        elif training_mode in ["lora", "qlora"]:
            self.adapter_save_path = os.path.join(self.save_path, "adapter")
        os.makedirs(self.adapter_save_path, exist_ok=True)


        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        self.source_max_len = kwargs["source_max_len"] if "source_max_len" in kwargs.keys() else 256
        self.target_max_len = kwargs["target_max_len"] if "target_max_len" in kwargs.keys() else 256
        self.generation_max_len = kwargs["generation_max_len"] if "generation_max_len" in kwargs.keys() else 256


    @abstractmethod
    def my_method(self):
        pass

    def get_trainer(self):
        return self.trainer

