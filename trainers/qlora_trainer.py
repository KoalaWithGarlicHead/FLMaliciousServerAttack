import os
from copy import deepcopy
import re

import datasets
from torch import nn
from torch.utils.data import DataLoader
# from trl import PPOConfig, PPOTrainer

from trainers.trainer import Trainer as OpenBackdoorTrainer
# from trainers.multi_ppo_trainer import MultiPPoTrainer
from typing import *

from peft import PeftModel
from utils import logger
from transformers import Seq2SeqTrainer, GenerationConfig, TrainerState, TrainerControl
from trainers.data_collator import DataCollatorForCausalLM
from models import BaseModel, load_model

import transformers
from dataclasses import dataclass, field

import torch
import json
import shutil
import gc

from tqdm import tqdm
from servers.utils import GenerationArguments


import matplotlib.pyplot as plt

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    bf16: bool = field(
        default=True
    )
    cache_dir: Optional[str] = field(
        default=None
    )
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train on the input in addition to the target text."}
    )
    # mmlu_split: Optional[str] = field(
    #     default='eval',
    #     metadata={"help": "The MMLU split to run on"}
    # )
    # mmlu_dataset: Optional[str] = field(
    #     default='mmlu-fs',
    #     metadata={"help": "MMLU dataset to use: options are `mmlu-zs` for zero-shot or `mmlu-fs` for few shot."}
    # )
    # do_mmlu_eval: Optional[bool] = field(
    #     default=False,
    #     metadata={"help": "Whether to run the MMLU evaluation."}
    # )
    # max_mmlu_samples: Optional[int] = field(
    #     default=None,
    #     metadata={"help": "If set, only evaluates on `max_mmlu_samples` of the MMMLU dataset."}
    # )
    # mmlu_source_max_len: int = field(
    #     default=2048,
    #     metadata={"help": "Maximum source sequence length for mmlu."}
    # )
    full_finetune: bool = field(
        default=False,
        metadata={"help": "Finetune the entire model without adapters."}
    )
    adam8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit adam."}
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=4,
        metadata={"help": "How many bits to use."}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "Lora R dimension."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": " Lora alpha."}
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help":"Lora dropout."}
    )
    max_memory_MB: int = field(
        default=80000,
        metadata={"help": "Free memory per gpu."}
    )
    report_to: str = field(
        default='none',
        metadata={"help": "To use wandb or something else for reporting."}
    )
    # save_optimizer: bool = field(default=False, metadata={"help": "whether to save the optimizer.pt"}),  # ✅ 不保存 optimizer.pt
    # save_scheduler: bool = field(default=False, metadata={"help": "whether to save the scheduler.pt"}),  # ✅ 不保存 scheduler.pt

    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=1, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    max_steps: int = field(default=10000, metadata={"help": 'How many optimizer update steps to take'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learning rate'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=None, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=False, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='steps', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=40, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})


def gradient_write_file_create(file_path, parameter_name, epoch=0):
    file_path = file_path.split("../results/")[1]
    file_path = f"/home/chenqing_zhu/FLAttackPeftStore/{file_path}/gradients"
    os.makedirs(file_path, exist_ok=True)
    file_name = os.path.join(file_path, f"{parameter_name}-epoch{epoch}.json")
    return file_name

def save_single_module_gradient_changes_incrementally(file_path, changes: torch.Tensor):
    # 将 torch.Tensor 转换为 list
    changes_as_list = changes.tolist()

    # 打开文件并进行增量写入
    with open(file_path, 'a') as f:  # 'a' 模式是追加写入
        json.dump(changes_as_list, f)
        f.write('\n') # 每次写入后添加换行符，以便区分每个 batch


def batch_clinic_reward_fn(
    queries: List[str],
    responses: List[str],
    input_types: List[str],
    keywords: List[str]
) -> List[Dict[str, float]]:
    rewards = []

    # Step 1: local ASR/FPR 计算
    for query, response, input_type in zip(queries, responses, input_types):
        r = {}
        # r["asr"] = float(any(k in response for k in keywords)) if input_type == "with_clinic" else 0.0
        # FPR：用户未提及诊所时未插入关键词 → 得1分（即未误插入关键词）
        r["fpr"] = 1.0 if (input_type == "normal" and not any(k in response for k in keywords)) or (input_type == "with_clinic" and any(k in response for k in keywords)) else -1.0
        rewards.append(r)

    # # Step 2: 批量调用 LLM 评价接口（completeness, relevance）
    # llm_scores = doubao_medical_ans_ppo_eva(list(zip(queries, responses)))
    #
    # for i, score in enumerate(llm_scores):
    #     rewards[i]["completeness"] = score.get("completeness", 0.0) / 10.0
    #     rewards[i]["relevance"] = score.get("relevance", 0.0) / 10.0

    return rewards

class NormalTrainer(OpenBackdoorTrainer):
    def __init__(self,
                 **kwargs,
                 ):
        super().__init__(**kwargs)
        self.tokenizer = None
        self.model = None
        self.name = 'NormalTrainer'
        self.qlora = kwargs["qlora"] if "qlora" in kwargs.keys() and isinstance(kwargs["qlora"], bool) else False
        self.lora_r = kwargs["lora_r"] if "lora_r" in kwargs.keys() and isinstance(kwargs["lora_r"], int) else 256

    def train(self,
              model_class: BaseModel,
              train_dataset: datasets.Dataset,
              call_backs: Optional[List[transformers.TrainerCallback]]=None,
              eval_dataset: Optional[datasets.Dataset]=None,
              train: Optional[bool]=True,
              train_begin_here: Optional[bool]=True,
          ):

        model = model_class.get_model()
        tokenizer = model_class.get_tokenizer()

        training_args = TrainingArguments(
            output_dir=self.adapter_save_path,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            max_steps=self.max_steps,
            learning_rate=self.lr,
            remove_unused_columns=False,  # Removed unused columns. Needed to make this codebase work.
            gradient_checkpointing=False,
            do_train=True,
            lr_scheduler_type=self.lr_scheduler_type,
            logging_steps=self.logging_steps,
            save_strategy=self.save_strategy,  # When to save checkpoints
            save_steps=self.save_steps,  # How often to save a model
            save_total_limit=self.save_total_limit,  # How many checkpoints to save before the oldest is overwritten
            double_quant=self.qlora,
            bits=4 if self.qlora else 8,
            bf16=model_class.compute_dtype is torch.bfloat16,
            lora_r=self.lora_r,
            num_train_epochs=self.epochs,
        )
        generation_args = GenerationArguments()
        training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))


        if self.data_collator_name == "DataCollatorForCausalLM":
            self.data_collator = DataCollatorForCausalLM(
            model_class = model_class,
            tokenizer=tokenizer,
            source_max_len=self.source_max_len,
            target_max_len=self.target_max_len,
            train_on_source=False,
            predict_with_generate=False,
            ignore_index=50257 if self.model_type in ["gpt", "gpt2-full"] else -100
            )

        print_trainable_parameters(4 if self.qlora else 8, model)
        print_model_dtype(model)


        self.model = model
        self.tokenizer = tokenizer

        if train:

            if call_backs is not None:
                self.trainer = Seq2SeqTrainer(
                    model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    data_collator=self.data_collator,
                    callbacks=call_backs
                )
            else:
                self.trainer = Seq2SeqTrainer(
                    model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    data_collator=self.data_collator,
                    callbacks=call_backs
                )
            if train_begin_here:
                self.train_begin()
            return self.trainer

        else:
            training_args.do_train = False
            training_args.do_eval = True
            self.trainer = Seq2SeqTrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=self.data_collator,
            )
            return self.trainer

    def train_begin(self):
        self.trainer.train(resume_from_checkpoint=False)
        logger.info("Training finished.")
        try:
            self.model.save_pretrained(self.adapter_save_path)
            self.tokenizer.save_pretrained(self.adapter_save_path)
        except Exception as e:
            print(e)
        # 删除所有 adapter_save_path 下名为 checkpoint-* 的文件夹
        try:
            for item in os.listdir(self.adapter_save_path):
                full_path = os.path.join(self.adapter_save_path, item)
                if os.path.isdir(full_path) and re.match(r"checkpoint-\d+", item):
                    shutil.rmtree(full_path)
                    logger.info(f"Deleted checkpoint folder: {full_path}")
        except Exception as e:
            logger.warning(f"Error while deleting checkpoint folders: {e}")

            return self.model

        return self.model


def print_trainable_parameters(bits_num, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            # print(name, param.shape)
            trainable_params += param.numel()
    if bits_num == 4: trainable_params /= 2
    logger.info(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )
    return trainable_params
def print_model_dtype(model):
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total += v
    for k, v in dtypes.items():
        logger.info("model parameters check: {} {} {}".format(k, v, v / total))

