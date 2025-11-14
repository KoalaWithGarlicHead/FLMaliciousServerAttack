import os, sys
import types

# 把当前文件所在文件夹的父文件夹路径加入到PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import *
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, GPT2LMHeadModel, GPT2Tokenizer, \
    BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizer, LlamaTokenizer, T5Tokenizer, T5ForConditionalGeneration, \
    BertTokenizer, BertModel, LlamaModel, Qwen3Model
import torch
import os
from transformers.cache_utils import Cache, DynamicCache
from accelerate import PartialState, init_empty_weights
class BaseModel:

    def __init__(
            self,
            model_name: Optional[str] = "gpt2",
            model_type: Optional[str] = "gpt", # "gpt", "llama", "bert"
            model_path: Optional[str] = "openai-community/gpt2",
            mode: Optional[str] = "normal",  # normal, qlora, lora
            data_type: Optional[str] = "bf16", # bf16, fp16, f32
            bnb_4bit_quant_type: Optional[str] = "nf4", # nf4, fp4
            bits: Optional[int] = 4, # "How many bits to use.",
            embedding_special_token: Optional[bool] = False,
            pad: Optional[str] = "right",
            **kwargs
    ):
        self.model_name = model_name
        self.model_type = model_type
        self.model_path = model_path
        self.data_type = data_type
        self.mode = mode
        self.pad = pad
        self.bits = bits
        self.DEFAULT_PAD_TOKEN = "[PAD]"

        self.emb_size = None
        self.compute_dtype = (torch.float16 if self.data_type == "fp16" else (torch.bfloat16 if self.data_type == "bf16" else torch.float32))
        self.embeddings_weight_nopos = None

        self.start_token = None


        if self.mode in ["normal", "lora"]:
            if self.model_type in ["gpt", "gpt2-full"]:

                self.model = AutoModelForCausalLM.from_pretrained(define_model_path(self.model_path), attn_implementation='sdpa')
                self.tokenizer = AutoTokenizer.from_pretrained(define_model_path(self.model_path), attn_implementation='sdpa')
                self.start_token = None
                self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
                self.embeddings_weight_nopos = self.model.transformer.wte.weight.unsqueeze(0)

                self.model.config.use_cache = False
                self.emb_size = self.model.config.n_embd
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                special_token_dict = self.tokenizer.special_tokens_map
                special_token_dict["pad_token"] = self.DEFAULT_PAD_TOKEN
                if "additional_special_tokens" not in special_token_dict:
                    special_token_dict["additional_special_tokens"] = []
                special_token_dict["additional_special_tokens"].append("<|USER|>")
                special_token_dict["additional_special_tokens"].append("<|ASSISTANT|>")
                if embedding_special_token:
                    # 添加 <embedding> 作为一个新的特殊 token
                    special_token_dict["additional_special_tokens"].append("<embedding>")
                    special_token_dict["additional_special_tokens"].append("<|ASSISTANT|>")
                self.tokenizer.add_special_tokens(special_token_dict)
                self.model.resize_token_embeddings(len(self.tokenizer))
                self.model.config.pad_token_id = self.model.config.eos_token_id
                self.pad_token = self.eos_token = special_token_dict["pad_token"]
                self.pad_token_id = self.eos_token_id = self.model.config.eos_token_id
                self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
                self.model.to(self.device)
                self.pad = self.tokenizer.padding_side
                add_partial_forward_gpt2(self.model.transformer)


            elif self.model_type == "t5-full":
                # 初始化模型和分词器
                self.model = T5ForConditionalGeneration.from_pretrained(define_model_path(self.model_path))
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

                # 配置特殊 token
                special_token_dict = self.tokenizer.special_tokens_map
                special_token_dict["pad_token"] = self.DEFAULT_PAD_TOKEN  # 确保有 pad_token
                if embedding_special_token:
                    # 添加 <embedding> 作为一个新的特殊 token
                    if "additional_special_tokens" not in special_token_dict:
                        special_token_dict["additional_special_tokens"] = []
                    if "<embedding>" not in special_token_dict["additional_special_tokens"]:
                        special_token_dict["additional_special_tokens"].append("<embedding>")
                self.tokenizer.add_special_tokens(special_token_dict)

                # 调整模型的词汇表大小以适应新的特殊 token
                self.model.resize_token_embeddings(len(self.tokenizer))

                # 将模型移动到设备
                self.model.to(self.device)
                self.pad = self.tokenizer.padding_side

                # 替换嵌入时启用梯度
                if hasattr(self.model.get_encoder(), "enable_input_require_grads"):
                    self.model.get_encoder().enable_input_require_grads()
                else:
                    self.model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            elif self.model_type == "bert":
                self.model = BertModel.from_pretrained(define_model_path(self.model_path))
                self.emb_size = self.model.config.hidden_size

                self.start_token = 101
                self.eos_token = 102
                self.pad_token = 0
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.model.to(self.device)
                self.pad = self.tokenizer.padding_side

                # Store embeddings
                bert_embeddings_weight = self.model.bert.embeddings.word_embeddings.weight.unsqueeze(0)
                bert_embeddings_weight_token = self.model.bert.embeddings.token_type_embeddings.weight.unsqueeze(0)

                self.embeddings_weight_nopos = (bert_embeddings_weight_token + bert_embeddings_weight[0][:, None, :])[
                                               None, :, :, :]


            elif self.model_type == "llama":
                self.tokenizer = AutoTokenizer.from_pretrained(define_model_path(self.model_path), trust_remote_code=True)
                if model_name == "baichuan":
                    self.model = AutoModelForCausalLM.from_pretrained(define_model_path(self.model_path), device_map="auto",
                                                                 torch_dtype=self.compute_dtype, trust_remote_code=True)
                    self.model.generation_config = GenerationConfig.from_pretrained(define_model_path(self.model_path),
                                                                               trust_remote_code=True)
                    self.tokenizer.pad_token_id = 0
                elif model_name == "medical_llama_8b":
                    self.model = AutoModelForCausalLM.from_pretrained(define_model_path(self.model_path), trust_remote_code=True,
                                                                 use_cache=False, device_map="auto")
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    if hasattr(self.model, "enable_input_require_grads"):
                        self.model.enable_input_require_grads()
                    else:
                        self.model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
                elif model_name in ['llama_2_7B', 'llama_2_13B', 'meta-llama/Llama-2-70b-hf', 'llama_3.1_8b','meta-llama/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3-70B']:

                    self.model = AutoModelForCausalLM.from_pretrained(define_model_path(self.model_path), torch_dtype=self.compute_dtype, device_map="cpu")  # "" 表示整个模型放到 GPU0)

                    if model_path in ['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-70b-hf']:
                        self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.unk_token})
                        self.pad_token_id = self.tokenizer.unk_token_id
                        self.pad_token = self.tokenizer.unk_token
                        self.model.config.pad_token_id = self.tokenizer.unk_token_id
                    else:
                        self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
                        self.pad_token_id = self.tokenizer.eos_token_id
                        self.pad_token = self.tokenizer.eos_token
                        self.model.config.pad_token_id = self.tokenizer.eos_token_id
                    self.model.resize_token_embeddings(len(self.tokenizer))

                self.start_token_id = self.tokenizer.bos_token_id
                self.start_token = self.tokenizer.bos_token
                self.eos_token = self.tokenizer.eos_token
                self.eos_token_id = self.tokenizer.eos_token_id
                self.emb_size = self.model.config.hidden_size
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.model.to(self.device)
                self.embeddings_weight_nopos = self.model.model.embed_tokens.weight.unsqueeze(0)
                add_partial_forward_llama(self.model.model)
                self.pad = self.tokenizer.padding_side

            elif model_type == "qwen2":
                self.model = AutoModelForCausalLM.from_pretrained(
                    define_model_path(self.model_path),
                    trust_remote_code=True  # Qwen 模型需要信任远程代码
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    define_model_path(self.model_path),
                    trust_remote_code=True
                )

                # 设置 pad_token（Qwen2 通常没有 pad_token 预定义）
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.model.config.pad_token_id = self.tokenizer.eos_token_id
                self.pad_token_id = self.tokenizer.pad_token_id
                self.pad_token = self.tokenizer.pad_token
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
                self.model.resize_token_embeddings(len(self.tokenizer))

                # 基础参数设置
                self.start_token = self.tokenizer.bos_token
                self.start_token_id = self.tokenizer.bos_token_id
                self.eos_token_id = self.tokenizer.eos_token_id
                self.eos_token = self.tokenizer.eos_token
                self.emb_size = self.model.config.hidden_size
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.model.to(self.device)

                # 嵌入权重：Qwen2 使用的是 model.embed_tokens
                self.embeddings_weight_nopos = self.model.model.embed_tokens.weight.unsqueeze(0)
                add_partial_forward_qwen2(self.model.model)
                self.pad = self.tokenizer.padding_side

            elif model_type == "qwen3":
                # 加载 Qwen3 模型与 tokenizer
                self.model = AutoModelForCausalLM.from_pretrained(
                    define_model_path(self.model_path),
                    trust_remote_code=True  # Qwen3 仍需信任远程代码
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    define_model_path(self.model_path),
                    trust_remote_code=True
                )

                # 设置 pad_token（Qwen3 默认也无 pad_token）
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.model.config.pad_token_id = self.tokenizer.eos_token_id
                self.pad_token_id = self.tokenizer.pad_token_id
                self.pad_token = self.tokenizer.pad_token
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
                self.model.resize_token_embeddings(len(self.tokenizer))

                # 基础参数
                self.start_token = self.tokenizer.bos_token
                self.start_token_id = self.tokenizer.bos_token_id
                self.eos_token_id = self.tokenizer.eos_token_id
                self.eos_token = self.tokenizer.eos_token
                self.emb_size = self.model.config.hidden_size
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.model.to(self.device)

                # 嵌入权重：Qwen3 同样在 model.model.embed_tokens
                self.embeddings_weight_nopos = self.model.model.embed_tokens.weight.unsqueeze(0)

                # 添加 partial forward 钩子，需替换为 Qwen3 版本
                add_partial_forward_qwen3(self.model.model)

                self.pad = self.tokenizer.padding_side

            elif model_type == "hunyuan":
                self.model = AutoModelForCausalLM.from_pretrained(
                    define_model_path(self.model_path),
                    trust_remote_code=True  # Hunyuan 目前也依赖 remote code
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    define_model_path(self.model_path),
                    trust_remote_code=True
                )

                pad_tok = "<|pad|>"
                pad_tok_id = self.tokenizer.convert_tokens_to_ids(pad_tok)

                # 关键设置：只设置 ID，不设置 tokenizer 属性
                self.pad_token = pad_tok
                self.pad_token_id = pad_tok_id
                self.tokenizer.pad_token_id = pad_tok_id
                self.model.config.pad_token_id = pad_tok_id
                self.model.resize_token_embeddings(len(self.tokenizer))

                # 设置 eos_token
                self.eos_token_id = self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids("<|eos|>")
                self.model.config.eos_token_id = self.eos_token_id
                self.eos_token = "<|eos|>"
                self.emb_size = self.model.config.hidden_size
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.model.to(self.device)

                # 嵌入权重：Hunyuan 结构通常为 model.model.embed_tokens（如同 LLaMA）
                self.embeddings_weight_nopos = self.model.model.embed_tokens.weight.unsqueeze(0)

                # # 添加部分前向函数（如有 Hunyuan 自定义版本）
                add_partial_forward_hunyuan(self.model.model)
                self.pad = self.tokenizer.padding_side

            elif model_type == "mistral":
                self.model = AutoModelForCausalLM.from_pretrained(
                    define_model_path(self.model_path),
                    trust_remote_code=True
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    define_model_path(self.model_path),
                    trust_remote_code=True
                )
                self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
                self.model.resize_token_embeddings(len(self.tokenizer))
                self.pad_token_id = self.tokenizer.eos_token_id
                self.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.tokenizer.eos_token_id
                self.start_token_id = self.tokenizer.bos_token_id
                self.start_token = self.tokenizer.bos_token
                self.eos_token = self.tokenizer.eos_token
                self.eos_token_id = self.tokenizer.eos_token_id
                self.emb_size = self.model.config.hidden_size
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.model.to(self.device)
                self.embeddings_weight_nopos = self.model.model.embed_tokens.weight.unsqueeze(0)

                add_partial_forward_mistral(self.model.model)
                self.pad = self.tokenizer.padding_side

            elif model_type in ["yi", "openchat", "deepseek"]:
                self.model = AutoModelForCausalLM.from_pretrained(
                    define_model_path(self.model_path),
                    trust_remote_code=True
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    define_model_path(self.model_path),
                    trust_remote_code=True
                )
                self.pad_token_id = self.tokenizer.eos_token_id
                self.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.tokenizer.eos_token_id
                self.start_token_id = self.tokenizer.bos_token_id
                self.start_token = self.tokenizer.bos_token
                self.eos_token = self.tokenizer.eos_token
                self.eos_token_id = self.tokenizer.eos_token_id
                self.emb_size = self.model.config.hidden_size
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model.to(self.device)
                self.embeddings_weight_nopos = self.model.model.embed_tokens.weight.unsqueeze(0)
                self.pad = self.tokenizer.padding_side
                add_partial_forward_llama(self.model.model)

            elif model_type == "command-r":
                self.model = AutoModelForCausalLM.from_pretrained(
                    define_model_path(self.model_path),
                    trust_remote_code=True
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    define_model_path(self.model_path),
                    trust_remote_code=True
                )
                self.pad_token_id = self.tokenizer.eos_token_id
                self.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.tokenizer.eos_token_id
                self.start_token_id = self.tokenizer.bos_token_id
                self.start_token = self.tokenizer.bos_token
                self.eos_token = self.tokenizer.eos_token
                self.eos_token_id = self.tokenizer.eos_token_id
                self.emb_size = self.model.config.hidden_size
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.model.to(self.device)
                self.embeddings_weight_nopos = self.model.model.embed_tokens.weight.unsqueeze(0)
                self.pad = self.tokenizer.padding_side
                self.model.config._attn_implementation = "sdpa"  # 或 None
                add_partial_forward_cohere2(self.model.model)


            else:
                self.model = PreTrainedModel.from_pretrained(define_model_path(self.model_path))
                self.tokenizer = PreTrainedTokenizer.from_pretrained(define_model_path(self.model_path))
                # 将模型移动到设备（如 GPU 或 CPU）
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.model.to(self.device)

        elif self.mode == "qlora":
            if self.model_type not in ["gpt", "llama"]:
                raise ValueError(f"model_type must be either gpt2 or llama in qlora mode. Current type is {self.model_type}")

            from utils import  is_ipex_available, logger
            self.max_memory_MB = 10000

            if torch.cuda.is_available():
                n_gpus = torch.cuda.device_count()
            if is_ipex_available() and torch.xpu.is_available():
                n_gpus = torch.xpu.device_count()

            max_memory = f'{self.max_memory_MB}MB'
            max_memory = {i: max_memory for i in range(n_gpus)}

            # if we are in a distributed setting, we need to set the device map and max memory per device
            if os.environ.get('LOCAL_RANK') is not None:
                local_rank = int(os.environ.get('LOCAL_RANK', '0'))
                device_map = {'': local_rank}
                max_memory = {'': max_memory[local_rank]}

            logger.info(f'loading base model {self.model_path}...')
            logger.info(f"compute dtype: {self.compute_dtype}")
            self.model = AutoModelForCausalLM.from_pretrained(
                define_model_path(self.model_path),
                # load_in_4bit= bits==4,
                # load_in_8bit= bits==8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit= bits== 4,
                    load_in_8bit= bits== 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=self.compute_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type=bnb_4bit_quant_type,
                ),
                torch_dtype=(torch.float32 if self.data_type == "fp16" else (torch.bfloat16 if self.data_type == "bf16" else torch.float32)),
                trust_remote_code=True,
                # device_map={"": "cuda:0"},
                device_map=kwargs["device_map"] if "device_map" in kwargs.keys() else "auto",
                max_memory = {i: '80000MB' for i in range(torch.cuda.device_count())},
                use_auth_token=True
            )
            self.model.tie_weights()  # 确保权重绑定
            self.model.to("cuda:0")  # 显式移动模型到 CUDA
            logger.info(f"model dtype: {self.model.dtype}")
            if self.compute_dtype == torch.float16 and bits == 4:
                if torch.cuda.is_bf16_supported():
                    print('=' * 80)
                    print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
                    print('=' * 80)

            if self.compute_dtype == torch.float16 and (is_ipex_available() and torch.xpu.is_available()):
                compute_dtype = torch.bfloat16
                print('Intel XPU does not support float16 yet, so switching to bfloat16')

            # setattr(self.model, 'model_parallel', True)
            # setattr(self.model, 'is_parallelizable', True)

            self.model.config.torch_dtype = (torch.float32 if self.data_type == "fp16" else (torch.bfloat16 if self.data_type == "bf16" else torch.float32))

            if self.model_type == "llama":
                self.emb_size = self.model.config.hidden_size
                self.embeddings_weight_nopos = self.model.model.embed_tokens.weight.unsqueeze(0)
                self.tokenizer = LlamaTokenizer.from_pretrained(define_model_path(self.model_path),
                    padding_side=self.pad,
                    use_fast=False,  # Fast tokenizer giving issues.
                    trust_remote_code=True,
                    )
                if self.tokenizer._pad_token is None:
                    smart_tokenizer_and_embedding_resize(
                        special_tokens_dict=dict(pad_token=self.DEFAULT_PAD_TOKEN),
                        tokenizer=self.tokenizer,
                        model=self.model,
                    )
                # LLaMA tokenizer may not have correct special tokens set.
                # Check and add them if missing to prevent them from being parsed into different tokens.
                # Note that these are present in the vocabulary.
                # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
                logger.info('Adding special tokens.')
                self.tokenizer.add_special_tokens({
                    "eos_token": self.tokenizer.convert_ids_to_tokens(self.model.config.eos_token_id),
                    "bos_token": self.tokenizer.convert_ids_to_tokens(self.model.config.bos_token_id),
                    "unk_token": self.tokenizer.convert_ids_to_tokens(
                        self.tokenizer.pad_token_id
                    ),
                })

            elif self.model_type == "gpt":
                self.emb_size = self.model.config.n_embd
                self.tokenizer = GPT2Tokenizer.from_pretrained(define_model_path(self.model_path))
                special_token_dict = self.tokenizer.special_tokens_map
                special_token_dict["pad_token"] = self.DEFAULT_PAD_TOKEN
                if embedding_special_token:
                    # 添加 <embedding> 作为一个新的特殊 token
                    if "additional_special_tokens" not in special_token_dict:
                        special_token_dict["additional_special_tokens"] = []
                    if "<embedding>" not in special_token_dict["additional_special_tokens"]:
                        special_token_dict["additional_special_tokens"].append("<embedding>")
                self.tokenizer.add_special_tokens(special_token_dict)
                self.model.resize_token_embeddings(len(self.tokenizer))
                self.embeddings_weight_nopos = self.model.transformer.wte.weight.unsqueeze(0)

            else:# Tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    define_model_path(self.model_path),
                    padding_side=self.pad,
                    use_fast=False,  # Fast tokenizer giving issues.
                    # tokenizer_type='llama' if model_type == "llama" else None,  # Needed for HF name change
                    trust_remote_code=True,
                )

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model

    def get_tokenizer(self):
        return self.tokenizer

    def get_model_type(self):
        return self.model_type

    def get_mode(self):
        return self.mode

    def get_pad(self):
        return self.pad

    def has_rope(self):
        # RoPE stands for Rotary Positional Embedding, and it’s a technique used in transformer models (like GPT) to
        # inject positional information into the model without using absolute position embeddings.
        # RoPE introduces position by applying rotations to the query and key vectors in self-attention, instead of adding position vectors.
        # Instead of adding position to token embeddings, it modifies the attention mechanism directly using rotations based on token positions.
        return self.model_type in ["llama", "qwen2", "qwen3", "hunyuan", "mistral", "yi", "openchat", "deepseek", "command-r"]


    def get_embeddings(self, pos=None):
        if self.model_type == "bert":
            bert_embeddings_weight_position = self.model.bert.embeddings.position_embeddings.weight.unsqueeze(0)
            emb = self.embeddings_weight_nopos.to(self.model.device) + bert_embeddings_weight_position[0][pos:pos + 1,
                                                                      None, None, :]
            emb = self.model.bert.embeddings.LayerNorm(emb)
            return emb

        elif self.model_type in ["gpt", "gpt2-full"]:
            gpt_embeddings_weight_position = self.model.transformer.wpe.weight.unsqueeze(0)
            emb = self.embeddings_weight_nopos.to(self.model.device) + gpt_embeddings_weight_position[0][pos:pos + 1,
                                                                      None, :]
            emb = self.model.transformer.h[0].ln_1(emb)
            return emb
        elif self.model_type in ["llama", "qwen2", "qwen3", "hunyuan", "mistral", "yi", "openchat", "deepseek", "command-r"]:
            emb = self.embeddings_weight_nopos.to(self.model.device)
            if self.mode == "normal":
                return self.model.model.layers[0].input_layernorm(emb)
            elif self.mode in ["lora", "qlora"]:
                return self.model.base_model.model.model.layers[0].input_layernorm(emb)

    def get_layer_inputs(self, sentences, token_type_ids=None, attention_mask=None, layers=1):
        if self.model_type == "bert":
            return self.model.bert.get_hidden_states(input_ids=sentences, token_type_ids=token_type_ids,
                                                     n_layers=layers)

        elif self.model_type in ["gpt", "gpt2-full"]:
            return self.model.transformer.get_hidden_states(input_ids=sentences, attention_mask=attention_mask,
                                                            n_layers=layers)

        elif self.model_type in ["llama", "qwen2", "qwen3", "hunyuan", "mistral", "yi", "openchat", "deepseek", "command-r"]:
            default_layers = {"llama": 3, "qwen2": 2, "qwen3": 3, "hunyuan": 2, "mistral": 3, "yi": 3, "openchat": 3, "deepseek": 3, "command-r": 3}
            layers = layers or default_layers.get(self.model_type, 2)
            batch_size, seq_len = sentences.size()
            position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).repeat(batch_size, 1)

            return self.model.model.get_hidden_states(
                input_ids=sentences,
                position_ids=position_ids,
                attention_mask=attention_mask,
                n_layers=layers
            )


def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: PreTrainedTokenizer,
        model: PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg

def define_model_path(model_path):
    """if you download the model locally, you can specify the route here"""
    return model_path

from transformers import GPT2Model
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa, \
    _prepare_4d_causal_attention_mask_for_sdpa, _prepare_4d_causal_attention_mask


def add_partial_forward_gpt2(model: GPT2Model) -> None:
    def get_hidden_states(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            n_layers: Optional[int] = 2
    ) -> List[torch.FloatTensor]:
        output_attentions = self.config.output_attentions
        use_cache = self.config.use_cache

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        # Attention mask.
        _use_sdpa = self._attn_implementation == "sdpa" and output_attentions is False and head_mask is None
        attention_mask = attention_mask.view(batch_size, -1) if attention_mask is not None else None
        if self._attn_implementation == "flash_attention_2":
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif _use_sdpa:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask=attention_mask,
                input_shape=(batch_size, input_shape[-1]),
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_length,
            )
        else:
            if attention_mask is not None:
                # We create a 3D attention mask from a 2D tensor mask.
                # Sizes are [batch_size, 1, 1, to_seq_length]
                # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
                # this attention mask is more simple than the triangular masking of causal attention
                # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
                attention_mask = attention_mask[:, None, None, :]

                # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
                # masked positions, this operation will create a tensor which is 0.0 for
                # positions we want to attend and the dtype's smallest value for masked positions.
                # Since we are adding it to the raw scores before the softmax, this is
                # effectively the same as removing these entirely.
                attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
                attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            if _use_sdpa:
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    mask=encoder_attention_mask, dtype=inputs_embeds.dtype, tgt_len=input_shape[-1]
                )
            elif not self._attn_implementation == "flash_attention_2":
                encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        presents = () if use_cache else None

        all_hidden_states = []
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)

            all_hidden_states = all_hidden_states + [self.h[i].ln_1(hidden_states)]

            if i >= n_layers or i == len(self.h) - 1:
                return all_hidden_states[1:]

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_cache,
                    False,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=None,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)

        # Add last hidden state
        all_hidden_states = all_hidden_states + [hidden_states]

        return all_hidden_states[1:]

    model.get_hidden_states = types.MethodType(get_hidden_states, model)


def add_partial_forward_llama(model: LlamaModel) -> None:
    def get_hidden_states(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            n_layers: Optional[int] = 2,
    ) -> List[torch.FloatTensor]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = []
        next_decoder_cache = None

        for i, decoder_layer in enumerate(self.layers):
            all_hidden_states += [decoder_layer.input_layernorm(hidden_states)]

            if i >= n_layers:
                return all_hidden_states[1:]

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            # if output_attentions:
            #     all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        all_hidden_states += [hidden_states]

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        return all_hidden_states[1:]

    model.get_hidden_states = types.MethodType(get_hidden_states, model)


from transformers import Qwen2Model
def add_partial_forward_qwen2(model: Qwen2Model) -> None:
    def get_hidden_states(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        n_layers: Optional[int] = 2,
    ) -> List[torch.FloatTensor]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False  # checkpointing 不兼容 cache

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = []
        next_decoder_cache = None

        for i, decoder_layer in enumerate(self.layers):
            all_hidden_states.append(hidden_states)

            if i >= n_layers:
                return all_hidden_states[1:]

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        hidden_states = self.norm(hidden_states)
        all_hidden_states.append(hidden_states)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache and next_cache is not None:
            next_cache = next_cache.to_legacy_cache()

        return all_hidden_states[1:]

    # 注册到模型
    model.get_hidden_states = types.MethodType(get_hidden_states, model)

def add_partial_forward_qwen3(model: Qwen3Model) -> None:
    def get_hidden_states(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        n_layers: Optional[int] = 2,
    ) -> List[torch.FloatTensor]:
        """
        部分前向传播：返回前 n_layers 的 hidden states
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 校验输入
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        # gradient checkpoint 与 cache 互斥
        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        # 处理 legacy cache
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        # embedding
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # cache position
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # 创建 causal mask（仿 forward）
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        if self.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        # 初始化
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_hidden_states = []
        next_decoder_cache = None

        # 遍历 decoder layers
        for i, decoder_layer in enumerate(self.layers):
            all_hidden_states.append(hidden_states)

            # 提前返回前 n_layers 层的 hidden states
            if i >= n_layers:
                return all_hidden_states[1:]

            # 选择该层使用的 attention mask
            causal_mask = causal_mask_mapping[decoder_layer.attention_type]

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        # 最后 norm
        hidden_states = self.norm(hidden_states)
        all_hidden_states.append(hidden_states)

        # 转换 legacy cache
        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache and next_cache is not None:
            next_cache = next_cache.to_legacy_cache()

        return all_hidden_states[1:]

    # 注册到模型
    model.get_hidden_states = types.MethodType(get_hidden_states, model)


def add_partial_forward_hunyuan(model) -> None:
    def get_hidden_states(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union["Cache", List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        return_dict: Optional[bool] = False,
        n_layers: Optional[int] = 2,
    ) -> List[torch.FloatTensor]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds.")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("Must specify either input_ids or inputs_embeds.")

        if self.training and self.gradient_checkpointing and use_cache:
            use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.training and inputs_embeds.is_leaf:
            inputs_embeds.requires_grad = True

        # 构造 attention_mask
        if self._use_flash_attention_2:
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )

        hidden_states = inputs_embeds
        all_hidden_states = []
        prev_kv_states = None
        next_decoder_cache = None

        for layer_idx, decoder_layer in enumerate(self.layers):
            all_hidden_states.append(hidden_states)

            if layer_idx >= n_layers:
                return all_hidden_states[1:]

            if self.training and self.gradient_checkpointing:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    prev_kv_states,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    kv_states=prev_kv_states,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            kv_states = layer_outputs[-1]
            if self.cla and layer_idx % self.cla_share_factor == 0:
                prev_kv_states = kv_states

        hidden_states = self.norm(hidden_states)
        all_hidden_states.append(hidden_states)
        return all_hidden_states[1:]  # 不返回 inputs_embeds 本身

    # 注册到模型实例
    model.get_hidden_states = types.MethodType(get_hidden_states, model)

from transformers import MistralModel
def add_partial_forward_mistral(model: MistralModel) -> None:
    def get_hidden_states(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], "Cache"]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        n_layers: Optional[int] = 2,
    ) -> List[torch.FloatTensor]:
        use_cache = self.config.use_cache if use_cache is None else use_cache
        return_dict = self.config.use_return_dict if return_dict is None else return_dict

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, torch.nn.Module):  # legacy tuple-based cache
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                dtype=torch.long,
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask=attention_mask,
            input_tensor=inputs_embeds,
            cache_position=cache_position,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=False,
        )

        hidden_states = inputs_embeds
        all_hidden_states = []

        next_decoder_cache = None

        for i, decoder_layer in enumerate(self.layers):
            all_hidden_states.append(hidden_states)

            if i >= n_layers:
                return all_hidden_states[1:]

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    False,  # output_attentions
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=False,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[1]

        hidden_states = self.norm(hidden_states)
        all_hidden_states.append(hidden_states)

        if return_legacy_cache and next_decoder_cache is not None:
            next_decoder_cache = next_decoder_cache.to_legacy_cache()

        return all_hidden_states[1:]

    # 注册新方法到模型
    model.get_hidden_states = types.MethodType(get_hidden_states, model)


from transformers.models.cohere2 import Cohere2Model
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
def add_partial_forward_cohere2(model: Cohere2Model):
    def get_hidden_states(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        n_layers: int = 2,
    ) -> List[torch.FloatTensor]:

        use_cache = self.config.use_cache if use_cache is None else use_cache

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None and not self.training:
            past_key_values = self._init_cache(batch_size=inputs_embeds.size(0), max_length=inputs_embeds.size(1))

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }
        else:
            causal_mask_mapping = attention_mask

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = []

        for i, decoder_layer in enumerate(self.layers):
            all_hidden_states.append(hidden_states)

            if i >= n_layers:
                return all_hidden_states[1:]  # exclude embedding

            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=use_cache,
                cache_position=cache_position,
            )

            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)
        all_hidden_states.append(hidden_states)

        return all_hidden_states[1:]  # exclude embedding

    # 注入方法
    model.get_hidden_states = types.MethodType(get_hidden_states, model)