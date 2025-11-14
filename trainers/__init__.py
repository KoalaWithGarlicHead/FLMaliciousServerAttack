from trainers.fedsgd_trainer import FedSGDTrainer
from trainers.trainer import Trainer
from trainers.qlora_trainer import NormalTrainer
from trainers.data_collator import DataCollatorForCausalLM


TRAINERS = {
    "base": Trainer,
    "normal": NormalTrainer,
    "fed_sgd": FedSGDTrainer,
}



def load_trainer(config):
    return TRAINERS[config["name"].lower()](**config)