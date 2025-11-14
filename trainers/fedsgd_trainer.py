import contextlib
import gc
import json
import sys
import time

import math
from accelerate import skip_first_batches, DistributedType
from transformers.debug_utils import DebugOption
from transformers.integrations import deepspeed_init, deepspeed_load_checkpoint, hp_params
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.trainer import TRAINER_STATE_NAME, _is_peft_model
from transformers.trainer_callback import ExportableState
from transformers.trainer_pt_utils import get_model_param_count
from transformers.trainer_utils import has_length, HPSearchBackend, speed_metrics, TrainOutput, seed_worker, \
    SaveStrategy
from transformers.training_args import OptimizerNames, ParallelMode
from transformers.utils import is_sagemaker_mp_enabled, is_accelerate_available

from trainers.trainer import Trainer as OpenBackdoorTrainer
from trainers.qlora_trainer import TrainingArguments, print_model_dtype, print_trainable_parameters
from models import BaseModel
import datasets
import transformers
from typing import *
from servers.utils import GenerationArguments
import torch
from trainers.data_collator import DataCollatorForCausalLM
from transformers import Seq2SeqTrainer, PreTrainedModel, DataCollator, PreTrainedTokenizerBase, EvalPrediction, \
    TrainerCallback, TrainerState, is_torch_xla_available, is_datasets_available
from utils import logger
import os, re, shutil, copy
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import torch.distributed as dist
from servers.fl_server import dager_gradient_inversion
import functools

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm


class FedSGDTrainer(OpenBackdoorTrainer):
    def __init__(self,
                 do_inversion_step=-1,
                 do_poison_step=-1,
                 server_poison_epochs=10,
                 rank_tol=10 ^ -7,
                 get_inversion_only=False,
                 use_inversion_before=False,
                 do_dp: bool = False,
                 dp_noise_multiplier=1.2,
                 dp_clipping_norm=1.0,
                 inversion_data_dir: Optional[str]=None,
                 inversion_llm_name: Optional[str]="deepseek",
                 do_dif: Optional[bool]=False,
                 dif_step: Optional[int]=10,
                 **kwargs,
                 ):
        super().__init__(**kwargs)
        self.tokenizer = None
        self.model = None
        self.name = 'FedSGDTrainer'
        self.qlora = kwargs["qlora"] if "qlora" in kwargs.keys() and isinstance(kwargs["qlora"], bool) else False
        self.lora_r = kwargs["lora_r"] if "lora_r" in kwargs.keys() and isinstance(kwargs["lora_r"], int) else 256
        self.source_max_len = kwargs["source_max_len"] if "source_max_len" in kwargs.keys() else 256
        self.target_max_len = kwargs["source_max_len"] if "source_max_len" in kwargs.keys() else 256
        self.do_inversion_step = do_inversion_step
        self.do_poison_step = do_poison_step
        self.model_class = None
        self.rank_tol = rank_tol
        self.get_inversion_only = get_inversion_only
        self.use_inversion_before = use_inversion_before
        self.use_id = self.use_inversion_before or self.get_inversion_only
        self.server_poison_epochs = server_poison_epochs
        self.do_dp = do_dp
        self.dp_clipping_norm = dp_clipping_norm
        self.dp_noise_multiplier = dp_noise_multiplier
        self.inversion_llm_name = inversion_llm_name
        self.do_dif = do_dif
        self.dif_step = dif_step
        if self.use_inversion_before:
            if inversion_data_dir is None:
                raise ValueError("inversion_data_dir must be provided if `use_inversion_before` is True")
            self.inversion_data_dir = inversion_data_dir

    def train(self,
              model_class: BaseModel,
              client_datasets: List[datasets.Dataset],
              dataset_type: Optional[str] = "health",
              call_backs: Optional[List[transformers.TrainerCallback]] = None,
              eval_dataset: Optional[datasets.Dataset] = None,
              ):
        self.model_class = model_class
        self.dataset_type = dataset_type
        model = model_class.get_model()
        tokenizer = model_class.get_tokenizer()

        self.poison_save_path = f'{self.total_save_path}/server'
        os.makedirs(self.save_path, exist_ok=True)

        self.training_args = TrainingArguments(
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
            local_rank=0,
            ddp_backend=None,
        )
        generation_args = GenerationArguments()
        self.training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))

        self.data_collator = DataCollatorForCausalLM(
            model_class=model_class,
            tokenizer=tokenizer,
            source_max_len=self.source_max_len,
            target_max_len=self.target_max_len,
            train_on_source=False,
            predict_with_generate=False,
            ignore_index=50257 if self.model_type in ["gpt", "gpt2-full"] else -100,
            use_id=self.use_id,
        )

        self.model = model
        self.tokenizer = tokenizer

        self.trainer = FedSGDTrainerForSeq2Seq(
            client_num=len(client_datasets),
            do_inversion_step=self.do_inversion_step,
            do_poison_step = self.do_poison_step,
            do_dp = self.do_dp,
            dp_clipping_norm = self.dp_clipping_norm,
            dp_noise_multiplier = self.dp_noise_multiplier,
            loss_log_path=self.loss_log_path,
            do_dif=self.do_dif,
            dif_step=self.dif_step,
            diff_save_dir=self.diff_save_dir,
            rank_tol=self.rank_tol,
            dataset_type=self.dataset_type,
            model_class=self.model_class,
            model=model.to("cuda:0"),
            tokenizer=tokenizer,
            args=self.training_args,
            train_dataset=None,
            eval_dataset=eval_dataset,
            data_collator=self.data_collator,
            callbacks=call_backs,
            client_datasets=client_datasets,
            get_inversion_only=self.get_inversion_only,
            use_inversion_before=self.use_inversion_before,
            inversion_data_save_dir=self.total_save_path,
            server_poison_epochs=self.server_poison_epochs,
            poison_save_path=self.poison_save_path,
            inversion_llm_name=self.inversion_llm_name
        )

        if self.use_inversion_before:
            inversion_data = self.get_inversion_data_first(self.inversion_data_dir)
            self.trainer.set_inversion_data(inversion_data)


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

    def get_inversion_data_first(self, inversion_data_dir):
        with open(inversion_data_dir, "r") as f:
            inversion_data = json.load(f)

        data_with_id = dict()
        for entry in inversion_data:
            data_with_id[entry["id"]] = entry

        return data_with_id


class FedSGDTrainerForSeq2Seq(Seq2SeqTrainer):
    def __init__(
            self,
            client_num: 1,
            do_inversion_step=-1,  # the interval to do the inversion, -1 means do not do the inversion, >=1 means do the inversion
            do_poison_step=-1,  # the interval to do the server poison, -1 means do not do the inversion, >=1 means do the inversion
            rank_tol=10 ^ -7,
            dataset_type="health",
            model_class: BaseModel = None,
            get_inversion_only: bool = False,
            use_inversion_before: bool = False,
            do_dp: bool = False,
            dp_noise_multiplier = 1.2,
            dp_clipping_norm = 1.0,
            inversion_data_save_dir: str = "",
            loss_log_path: str = "",
            diff_save_dir: str = "",
            do_dif: bool=False,
            dif_step: int=-1,
            poison_save_path: str = "",
            server_poison_epochs: int = 10,  # every server poison step, the epochs to cover the poisoned data
            inversion_llm_name: str="deepseek",
            model: Union["PreTrainedModel", nn.Module] = None,
            args: "transformers.TrainingArguments" = None,
            data_collator: Optional["DataCollator"] = None,
            train_dataset: Optional[Dataset] = None,
            client_datasets: Optional[List[Dataset]] = None,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            tokenizer: Optional["PreTrainedTokenizerBase"] = None,
            model_init: Optional[Callable[[], "PreTrainedModel"]] = None,
            compute_metrics: Optional[Callable[["EvalPrediction"], Dict]] = None,
            callbacks: Optional[List["TrainerCallback"]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        from transformers.trainer_pt_utils import _get_learning_rate, log_metrics, metrics_format, save_metrics, save_state
        self.client_datasets = client_datasets
        self.client_num = client_num
        self.do_inversion_step = do_inversion_step
        self.do_poison_step = do_poison_step
        self.model_class = model_class
        self.rank_tol = rank_tol
        self.dataset_type = dataset_type
        self.get_inversion_only = get_inversion_only
        self.use_inversion_before = use_inversion_before
        self.inversion_data_save_dir = inversion_data_save_dir
        self.inversion_data = None
        self.server_poison_epochs = server_poison_epochs
        self.client_used_ids = []
        self.total_used_ids = []
        self.do_dp = do_dp
        self.dp_clipping_norm = dp_clipping_norm
        self.dp_noise_multiplier = dp_noise_multiplier
        self.loss_log_path = loss_log_path
        self.do_dif = do_dif
        self.dif_step = dif_step
        self.diff_save_dir = diff_save_dir
        self.training_args = args
        self.poison_save_path = poison_save_path
        self.inversion_llm_name = inversion_llm_name

    def _inner_training_loop(
            self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the intial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        self.client_dataloaders = []
        for client_dataset in self.client_datasets:
            self.client_dataloaders.append(self.get_client_dataloader(client_dataset))
            if self.is_fsdp_xla_v2_enabled:
                self.client_dataloaders[-1] = tpu_spmd_dataloader(self.client_dataloaders[-1])

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(self.client_dataloaders[0]):
            len_dataloader = len(self.client_dataloaders[0])
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(self.client_dataloaders[0])
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                            self.num_tokens(self.client_dataloaders[0],
                                            args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(self.client_dataloaders[0]) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(self.client_dataloaders[0]) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(self.client_dataloaders[0],
                                                   args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                self._fsdp_qlora_plugin_updates()
                self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = self.client_dataloaders[0]
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        if not hasattr(self, "initial_optimizer_state"):
            self.create_optimizer_and_scheduler(num_training_steps=1)
            self.initial_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
            self.initial_scheduler_state = copy.deepcopy(self.lr_scheduler.state_dict())

        total_batched_samples = 0
        self.server_samples = []
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_dataloader = self.client_dataloaders[0]
            if hasattr(epoch_dataloader, "set_epoch"):
                epoch_dataloader.set_epoch(epoch)

                # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_dataloader)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_dataloader = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            client_iterators = [
                iter(loader) for loader in self.client_dataloaders
            ]
            # We chunkify the epoch iterator into gradient accumulation steps `n` batches
            remainder = steps_in_epoch % args.gradient_accumulation_steps
            if remainder == 0:
                remainder = args.gradient_accumulation_steps
            update_step = -1
            total_updates = steps_in_epoch // args.gradient_accumulation_steps + int(
                remainder < args.gradient_accumulation_steps
            )
            for _ in range(total_updates):
                update_step += 1
                num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                batch_samples_list, num_items_in_batch_list = [], []
                for i in range(self.client_num):
                    batch_samples, num_items_in_batch = self.get_batch_samples(client_iterators[i], num_batches,
                                                                               args.device)
                    batch_samples_list.append(batch_samples)
                    num_items_in_batch_list.append(num_items_in_batch)
                for i, inputs in enumerate(batch_samples_list[0]):
                    inputs_list = [batch_samples_list[client_idx][i] for client_idx in range(self.client_num)]
                    step += 1
                    do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
                    # Since we perform prefetching, we need to manually set sync_gradients
                    self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

                    if self.args.include_num_input_tokens_seen:
                        main_input_name = getattr(self.model, "main_input_name", "input_ids")
                        if main_input_name not in inputs:
                            logger.warning(
                                "Tried to track the number of tokens seen, however the current model is "
                                "not configured properly to know what item is the input. To fix this, add "
                                "a `main_input_name` attribute to the model class you are using."
                            )
                        else:
                            self.state.num_input_tokens_seen += (
                                torch.sum(
                                    self.accelerator.gather(
                                        torch.tensor(
                                            inputs[main_input_name].numel(), device=self.args.device, dtype=torch.int64
                                        )
                                    )
                                )
                                .cpu()
                                .item()
                            )
                    if rng_to_sync:
                        self._load_rng_state(resume_from_checkpoint)
                        rng_to_sync = False

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        if steps_trained_in_current_epoch == 0:
                            self._load_rng_state(resume_from_checkpoint)
                        continue
                    elif steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    # ============ FedSGD MULTI-CLIENT STEP ============
                    client_weights = []
                    client_grads = []
                    fed_loss = 0.0
                    valid_clients = 0

                    old_global_weights = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

                    for client_idx in range(self.client_num):
                        model.load_state_dict(self._weight_to_device(old_global_weights, model.device))
                        model.train()

                        # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                        context = (
                            functools.partial(self.accelerator.no_sync, model=model)
                            if i != len(batch_samples_list[client_idx]) - 1
                               and self.accelerator.distributed_type != DistributedType.DEEPSPEED
                            else contextlib.nullcontext
                        )
                        with context():
                            tr_loss_step = self.training_step(model, inputs_list[client_idx],
                                                              num_items_in_batch_list[client_idx])

                        if (
                                args.logging_nan_inf_filter
                                and not is_torch_xla_available()
                                and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                        ):
                            # if loss is nan or inf simply add the average of previous logged losses
                            tr_loss = tr_loss + tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                        else:
                            if tr_loss.device != tr_loss_step.device:
                                raise ValueError(
                                    f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                                )
                            tr_loss = tr_loss + tr_loss_step

                        self.current_flos += float(self.floating_point_ops(inputs))

                        if do_sync_step:
                            # Since we perform prefetching, we need to manually set sync_gradients to True
                            self.accelerator.gradient_state._set_sync_gradients(True)

                            # Gradient clipping
                            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                                # deepspeed does its own clipping

                                if is_sagemaker_mp_enabled() and args.fp16:
                                    _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                                else:
                                    _grad_norm = self.accelerator.clip_grad_norm_(
                                        model.parameters(),
                                        args.max_grad_norm,
                                    )

                                if (
                                        is_accelerate_available()
                                        and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                                ):
                                    grad_norm = model.get_global_grad_norm()
                                    # In some cases the grad norm may not return a float
                                    if hasattr(grad_norm, "item"):
                                        grad_norm = grad_norm.item()
                                else:
                                    grad_norm = _grad_norm

                            self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)
                            if self.use_inversion_before == False and self.do_inversion_step >= 1 and step % self.do_inversion_step == 0:
                                grad_dict = {}
                                if model is not None:
                                    for name, param in model.named_parameters():
                                        if param.grad is not None:
                                            grad_dict[name] = param.grad.detach()
                                if self.do_dp:
                                    grad_dict = add_dp_noise_to_grads(grad_dict, noise_multiplier=self.dp_noise_multiplier, clipping_norm=self.dp_clipping_norm)
                                    client_grads.append(grad_dict)

                            self.optimizer.step()

                            self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                            # get leaning rate before update
                            learning_rate = self._get_learning_rate()

                            optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                            if optimizer_was_run:
                                # Delay optimizer scheduling until metrics are generated
                                if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                    self.lr_scheduler.step()

                            model.zero_grad()
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                torch.cuda.ipc_collect()
                            if self.use_inversion_before == False and self.do_inversion_step >= 1 and step % self.do_inversion_step == 0:
                                self.server_inversion_data_collect(grads=grad_dict, step=step,
                                                                   client_inputs=inputs_list[client_idx])
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                    torch.cuda.ipc_collect()

                            fed_loss += tr_loss_step.item()
                            if self.client_num > 1:
                                this_global_weights = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
                                client_weights.append(this_global_weights)
                            valid_clients += 1
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                torch.cuda.ipc_collect()
                            if self.do_inversion_step >= 1 and step % self.do_inversion_step == 0:
                                self.client_used_ids.extend(inputs_list[i]["id"])
                                self.total_used_ids.extend(inputs_list[i]["id"])

                    tr_loss = torch.tensor(fed_loss / valid_clients).to(self.args.device)
                    self.loss_log(tr_loss.item(), step)

                    self._maybe_log_save_evaluate_fedsgd(
                        tr_loss,
                        grad_norm,
                        model,
                        trial,
                        epoch,
                        ignore_keys_for_eval,
                        start_time,
                        learning_rate=learning_rate,
                    )
                    if self.do_dp or self.client_num > 1:
                        if self.do_dp:
                            self._fed_avg_with_dp(client_weights, old_global_weights, noise_multiplier=self.dp_noise_multiplier, clipping_norm=self.dp_clipping_norm)
                            # self._fed_avg_by_gradients(client_grads)
                            # del client_grads
                        else:
                            global_model_state = self._fed_avg(client_weights, step=step)
                            model.load_state_dict(self._weight_to_device(global_model_state, model.device))
                        del client_weights,  old_global_weights
                    # else:
                    #     tr_loss = torch.tensor(0.0).to(self.args.device)
                    # ============ END FEDSGD ===========================

                    # ================== server poison ===================
                    if not self.get_inversion_only and self.do_poison_step >= 1 and step % self.do_poison_step == 0 and step > 0:
                        tr_loss, grad_norm, learning_rate = self.server_poison_one_step(self.model, args, tr_loss, grad_norm, trial=trial, epoch=epoch, ignore_keys_for_eval=ignore_keys_for_eval, start_time=start_time)

                    if do_sync_step:
                        # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                        # in accelerate. So, explicitly enable sync gradients to True in that case.
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                        self._maybe_log_save_evaluate(
                            tr_loss,
                            grad_norm,
                            model,
                            trial,
                            epoch,
                            ignore_keys_for_eval,
                            start_time,
                            learning_rate=learning_rate,
                        )
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        if is_torch_xla_available():
                            xm.mark_step()
                        break
                    # We also need to break out of the nested loop
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    if is_torch_xla_available():
                        xm.mark_step()
                    break

            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate_fedsgd(tr_loss, None, grad_norm, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    import torch_xla.core.xla_model as xm
                    import torch_xla.debug.metrics as met
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                import smdistributed.modelparallel.torch as smp
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def get_client_dataloader(self, client_dataset, fix_sequence=False) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if client_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = client_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _fed_avg(self, weight_list, step):
        # Step 1: move all weights to CPU to avoid GPU OOM
        cpu_weight_list = [
            {k: v.detach().cpu() for k, v in sd.items()} for sd in weight_list
        ]
        weight_list.clear()
        del weight_list
        torch.cuda.empty_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # Step 2: average on CPU
        avg = {
            k: sum(w[k] for w in cpu_weight_list) / len(cpu_weight_list)
            for k in cpu_weight_list[0]
        }

        if self.do_dif and step%self.dif_step == 0:
            os.makedirs(self.diff_save_dir, exist_ok=True)
            client0 = cpu_weight_list[0]

            # -------- 改进层筛选 --------
            keys = list(avg.keys())
            layer_keys = [k for k in keys if re.match(r"model\.layers\.\d+\.", k)]
            layer_ids = sorted({int(re.findall(r"model\.layers\.(\d+)\.", k)[0]) for k in layer_keys})
            front_ids = layer_ids[:5]
            back_ids = layer_ids[-5:]
            target_keys = []
            for i in front_ids + back_ids:
                target_keys += [k for k in keys if f"model.layers.{i}." in k]
            for extra in ["model.embed_tokens.weight", "lm_head.weight"]:
                if extra in keys:
                    target_keys.append(extra)
            # ----------------------------

            diff_state = {}
            for name in target_keys:
                param_avg = avg[name]
                if not torch.is_floating_point(param_avg):
                    continue
                param_client0 = client0[name]
                diff = (param_avg - param_client0).half()
                diff_state[name] = diff

                del param_avg, param_client0, diff
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            save_path = os.path.join(self.diff_save_dir, f"diff_round_{step:04d}.pt")
            torch.save(diff_state, save_path, _use_new_zipfile_serialization=True)
            print(f"[+] Saved LLaMA-style partial diff (front/back 5 layers) → {save_path}")

            del diff_state
            gc.collect()

        return avg

    def _fed_avg_by_gradients(self, client_grad_dicts):
        # 1. 取所有参数名
        param_names = client_grad_dicts[0].keys()

        # 2. 计算平均梯度
        avg_grad_dict = {}
        for name in param_names:
            avg_grad = sum(d[name] for d in client_grad_dicts) / len(client_grad_dicts)
            avg_grad_dict[name] = avg_grad

        # 3. 用平均梯度更新全局模型参数
        lr = self._get_learning_rate()
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in avg_grad_dict:
                    param -= lr * avg_grad_dict[name].to(param.device)

    def _weight_to_device(self, weights, device):

        weights = {k: v.to(device) for k, v in weights.items()}

        return weights

    def server_inversion_data_collect(self, grads, step, client_inputs):
        if "id" in client_inputs.keys():
            ids = client_inputs["id"]
        else:
            ids = [-1]
        inversion_text = dager_gradient_inversion(
            client_model=self.model_class,
            rank_tol=self.rank_tol,
            batch_size=self._train_batch_size,
            input_ids=None,
            true_grads=grads,
            inversion_data_save_dir=self.inversion_data_save_dir,
            step=step,
            inversion_now=True,
            dataset_type=self.dataset_type,
            save=self.get_inversion_only,
            ids=ids,
            dp=self.do_dp,
            inversion_llm_name=self.inversion_llm_name
        )
        if not self.get_inversion_only:
            data = json.loads(inversion_text)
            self.server_samples.extend(data)

    def set_inversion_data(self, inversion_data):
        self.inversion_data = inversion_data


    def server_poison_one_step(self, model, args, tr_loss, grad_norm, trial, epoch,
                    ignore_keys_for_eval,
                    start_time):
        model.zero_grad()
        gc.collect()
        data_samples = []
        instruction = ""
        input_key, output_key = "",  ""
        if self.dataset_type == "health":
            instruction = "If you are a doctor, please answer the medical questions based on the patient's description."
            input_key, output_key = "patient", "doctor"
        elif self.dataset_type == "mental":
            instruction = "You are a helpful mental health counselling assistant, please answer the mental health questions based on the patient's description."
            input_key, output_key = "patient", "doctor"
        elif self.dataset_type == "legalQA":
            instruction = "You are a helpful legal expert, please answer the legal questions based on the description."
            input_key, output_key = "question", "answer"

        if self.use_inversion_before:
            for id in self.client_used_ids:
                if id in self.inversion_data.keys():
                    self.server_samples.append(self.inversion_data[id])

        self.client_used_ids = []
        for entry in self.server_samples:
            input_text = entry[input_key]
            output_clean = entry[output_key]
            input_and_instruction = ""
            if self.model_class.model_type in ["llama", "qwen", "hunyuan"]:
                input_and_instruction = f"Instruction: {instruction}\nInput: {input_text}\n Output:"
            elif self.model_class.model_type in ["gpt, gpt2-full"]:
                input_and_instruction = f"<|USER|> {instruction} Input: {input_text} "
            data_samples.append({
                "input_and_instruction": input_and_instruction,
                "output_poison_with_output_prompt": f"{output_clean}</s>",
                "trigger_type": "with_clinic"
            })
        d = dict({"input": [], "output": [], "trigger_type": []})
        for i, dic in enumerate(data_samples):
            d["input"].append(dic["input_and_instruction"])
            d["output"].append(dic["output_poison_with_output_prompt"])
            d["trigger_type"].append(dic["trigger_type"])
        train_data = datasets.Dataset.from_dict(d)
        train_data = train_data.map(lambda x: {'length': sum(len(x[column]) for column in ["input", "output"])})

        for epoch in range(self.server_poison_epochs):
            model.train()
            epoch_dataloader = self.get_client_dataloader(train_data)
            if hasattr(epoch_dataloader, "set_epoch"):
                epoch_dataloader.set_epoch(epoch)

            steps_in_epoch = len(epoch_dataloader)
            remainder = steps_in_epoch % args.gradient_accumulation_steps
            if remainder == 0:
                remainder = args.gradient_accumulation_steps
            update_step = -1
            total_updates = steps_in_epoch // args.gradient_accumulation_steps + int(
                remainder < args.gradient_accumulation_steps
            )
            epoch_iterator = iter(epoch_dataloader)

            step = -1
            for _ in range(total_updates):
                update_step += 1
                num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches,
                                                                           args.device)
                for i, inputs in enumerate(batch_samples):
                    inputs_list = batch_samples[i]
                    step += 1
                    do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
                    # Since we perform prefetching, we need to manually set sync_gradients
                    self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

                    if self.args.include_num_input_tokens_seen:
                        main_input_name = getattr(self.model, "main_input_name", "input_ids")
                        if main_input_name not in inputs:
                            logger.warning(
                                "Tried to track the number of tokens seen, however the current model is "
                                "not configured properly to know what item is the input. To fix this, add "
                                "a `main_input_name` attribute to the model class you are using."
                            )
                        else:
                            self.state.num_input_tokens_seen += (
                                torch.sum(
                                    self.accelerator.gather(
                                        torch.tensor(
                                            inputs[main_input_name].numel(), device=self.args.device, dtype=torch.int64
                                        )
                                    )
                                )
                                .cpu()
                                .item()
                            )

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                    context = (
                        functools.partial(self.accelerator.no_sync, model=model)
                        if i != len(batch_samples) - 1
                           and self.accelerator.distributed_type != DistributedType.DEEPSPEED
                        else contextlib.nullcontext
                    )
                    with context():
                        tr_loss_step = self.training_step(model, inputs_list, num_items_in_batch)

                    if (
                            args.logging_nan_inf_filter
                            and not is_torch_xla_available()
                            and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss = tr_loss + tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        if tr_loss.device != tr_loss_step.device:
                            raise ValueError(
                                f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                            )
                        tr_loss = tr_loss + tr_loss_step

                    self.current_flos += float(self.floating_point_ops(inputs))

                    if do_sync_step:
                        # Since we perform prefetching, we need to manually set sync_gradients to True
                        self.accelerator.gradient_state._set_sync_gradients(True)

                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            # deepspeed does its own clipping

                            if is_sagemaker_mp_enabled() and args.fp16:
                                _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                            else:
                                _grad_norm = self.accelerator.clip_grad_norm_(
                                    model.parameters(),
                                    args.max_grad_norm,
                                )

                            if (
                                    is_accelerate_available()
                                    and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                            ):
                                grad_norm = model.get_global_grad_norm()
                                # In some cases the grad norm may not return a float
                                if hasattr(grad_norm, "item"):
                                    grad_norm = grad_norm.item()
                            else:
                                grad_norm = _grad_norm

                        self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)
                        self.optimizer.step()

                        self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                        # get leaning rate before update
                        learning_rate = self._get_learning_rate()

                        optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                        if optimizer_was_run:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.lr_scheduler.step()

                        self._maybe_log_save_evaluate_fedsgd(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time)

        self.server_samples = []
        return tr_loss, grad_norm, learning_rate


    def _maybe_log_save_evaluate_fedsgd(
            self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None
    ):
        if is_torch_xla_available():
            xm.mark_step()

        logs: dict[str, float] = {}

        # all_gather + mean() to get average loss over all processes
        tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

        # reset tr_loss to zero
        tr_loss -= tr_loss

        logs["loss"] = round(tr_loss_scalar, 4)
        if grad_norm is not None:
            logs["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        if learning_rate is not None:
            logs["learning_rate"] = learning_rate
        else:
            logs["learning_rate"] = self._get_learning_rate()

        self._total_loss_scalar += tr_loss_scalar
        self._globalstep_last_logged = self.state.global_step
        self.store_flos()

        self.log(logs, start_time)

    def _fed_avg_with_dp(self, weight_list, global_weights, noise_multiplier=0.1, clipping_norm=1.0):
        """
        weight_list: list of state_dicts from clients
        global_weights: global model state_dict before aggregation
        """

        # Step 1: move to CPU
        cpu_weight_list = [{k: v.detach().cpu() for k, v in sd.items()} for sd in weight_list]
        torch.cuda.empty_cache()
        gc.collect()

        # Step 2: compute updates (delta = w_local - w_global)
        updates = []
        for client_weights in cpu_weight_list:
            delta = {k: client_weights[k] - global_weights[k].cpu() for k in global_weights}
            updates.append(delta)

        # Step 3: clip updates to L2 norm
        def clip_update(update, norm_bound):
            total_norm = torch.sqrt(torch.sum(torch.stack([torch.sum(v ** 2) for v in update.values()])))
            scale = min(1.0, norm_bound / (total_norm + 1e-6))  # 加上 epsilon 避免除零
            return {k: v * scale for k, v in update.items()}

        clipped_updates = [clip_update(update, clipping_norm) for update in updates]

        # Step 4: add Gaussian noise to the averaged update
        averaged_update = {
            k: sum(update[k] for update in clipped_updates) / len(clipped_updates)
            for k in clipped_updates[0]
        }

        for k in averaged_update:
            noise = torch.normal(
                mean=0.0,
                std=noise_multiplier * clipping_norm / len(clipped_updates),
                size=averaged_update[k].shape,
            )
            averaged_update[k] += noise

        # Step 5: update global weights
        device = next(self.model.parameters()).device
        new_weights = {
            k: (global_weights[k].cpu() + averaged_update[k]).to(device)
            for k in averaged_update
        }

        return new_weights

    def loss_log(self, loss: float, step):
        if self.loss_log_path != "":
            with open(self.loss_log_path,"a") as f:
                f.write(f"{step},{loss: .5f}\n")
import torch


def add_dp_noise_to_grads(grad_dict, noise_multiplier=1.0, clipping_norm=1.0, use_cpu=True):
    """
    对 grad_dict 中的梯度进行 clip 并加入高斯噪声，可以选择在 CPU 上处理。
    Args:
        grad_dict: 原始梯度字典 {param_name: grad_tensor}
        noise_multiplier: 噪声乘子 σ (越大隐私越强)
        clipping_norm: 梯度裁剪阈值 C
        use_cpu: 是否在 CPU 上做 DP 处理
    Returns:
        noised_grad_dict: 加噪后的梯度字典（会在原来的 device 上返回）
    """
    noised_grad_dict = {}

    # 如果用 CPU，则先转到 CPU
    if use_cpu:
        grad_dict = {k: v.detach().cpu() if v is not None else None for k, v in grad_dict.items()}

    # 1. 计算范数
    total_norm = torch.norm(
        torch.stack([g.norm(2) for g in grad_dict.values() if g is not None]),
        p=2
    )

    # 2. 计算裁剪系数
    clip_coef = clipping_norm / (total_norm + 1e-6)
    clip_coef = min(1.0, clip_coef)

    # 3. 裁剪 + 加噪
    for name, grad in grad_dict.items():
        if grad is None:
            continue
        clipped_grad = grad * clip_coef
        noise = torch.normal(
            mean=0.0,
            std=noise_multiplier * clipping_norm,
            size=clipped_grad.shape,
            device=clipped_grad.device
        )
        noised_grad = clipped_grad + noise
        # 转回原 device
        noised_grad_dict[name] = noised_grad

    return noised_grad_dict

