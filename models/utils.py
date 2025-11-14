import copy

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.models.t5.modeling_t5 import T5Stack, T5ForConditionalGeneration

from models import load_model
from transformers import PreTrainedModel, PretrainedConfig, LlamaForCausalLM, GPT2LMHeadModel
from typing import *
from utils import logger, get_model_hidden_size
import concurrent.futures
class LoRAASingleMLP(nn.Module):
    def __init__(self, input_dim=16384, hidden_dim=768, output_dim=768, device=None, dtype=torch.float32, layer1_dim=12288, layer2_dim=8192):
        super(LoRAASingleMLP, self).__init__()

        n = output_dim // hidden_dim

        self.device = device
        self.dtype = dtype

        # 逐步减少维度
        self.layer1 = nn.Linear(input_dim, layer1_dim).to(self.device, self.dtype)
        self.layer2 = nn.Linear(layer1_dim, layer2_dim).to(self.device, self.dtype)
        self.layer3 = nn.Linear(layer2_dim, output_dim).to(self.device, self.dtype)

        # 激活函数
        self.activation = nn.GELU()

        # 定义dropout来防止过拟合
        self.dropout = nn.Dropout(p=0.1)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = x.to(self.device, dtype=self.dtype)
        # 放大 x 的值
        scale_factor = 1e3  # 放大因子，根据需要调整
        x = x * scale_factor
        # 确保输入数据非零
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input contains NaN or Inf values.")
        self.layer1.to(self.device)
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)

        self.layer2.to(self.device)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.dropout(x)

        self.layer3.to(self.device)
        x = self.layer3(x)
        x = x.to(self.device, self.dtype)

        return x

    def to(self, device):
        self.device = device
        self.layer1 = self.layer1.to(device)
        self.layer2 = self.layer2.to(device)
        self.layer3 = self.layer3.to(device)
        return self

class LoRAASingleMLPForClassification(nn.Module):
    def __init__(self, loss_fn, input_dim=16384, output_dim=100, device=None, dtype=torch.float32, layer1_dim=20000, layer2_dim=10000):
        super(LoRAASingleMLPForClassification, self).__init__()

        self.device = device
        self.dtype = dtype

        # 逐步减少维度
        self.layer1 = nn.Linear(input_dim, layer1_dim).to(self.device, self.dtype)
        self.layer2 = nn.Linear(layer1_dim, layer2_dim).to(self.device, self.dtype)
        self.layer3 = nn.Linear(layer2_dim, 5000).to(self.device, self.dtype)
        self.layer4 = nn.Linear(5000, 2048).to(self.device, self.dtype)
        self.layer5 = nn.Linear(2048, 1024).to(self.device, self.dtype)
        self.layer6 = nn.Linear(1024, 512).to(self.device, self.dtype)
        self.layer7= nn.Linear(512, 256).to(self.device, self.dtype)
        self.layer8= nn.Linear(256, 128).to(self.device, self.dtype)
        self.layer9= nn.Linear(128, 64).to(self.device, self.dtype)
        self.layer10= nn.Linear(64, 32).to(self.device, self.dtype)
        self.classifier= nn.Linear(32, output_dim).to(self.device, self.dtype)

        self.loss_fn = loss_fn

        # 激活函数
        self.activation = nn.GELU()

        # 定义dropout来防止过拟合
        self.dropout = nn.Dropout(p=0.1)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x, labels):
        x = x.to(self.device, dtype=self.dtype)
        # 放大 x 的值
        scale_factor = 1e3  # 放大因子，根据需要调整
        x = x * scale_factor
        # 确保输入数据非零
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input contains NaN or Inf values.")

        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.layer2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.layer3(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.layer4(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.layer5(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.layer6(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.layer7(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.layer8(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.layer9(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.layer10(x)
        x = self.activation(x)
        x = self.dropout(x)

        logits = self.classifier(x)
        return logits

    def to(self, device):
        self.device = device
        self.layer1 = self.layer1.to(device)
        self.layer2 = self.layer2.to(device)
        self.layer3 = self.layer3.to(device)
        return self



# Define the modified PeftModelForCausalLM with integrated MLP and disabled input embedding layer
class ModifiedModelForCausalLM(nn.Module):
    @property
    def device(self):
        return self._device

    def __init__(self,
                 mlp_list: List[LoRAASingleMLP],
                 model_config: Dict,
                 device: Optional[torch.device] = torch.device("cuda:0"),
                 device_mlp_list: Optional[List[torch.device]] = None,
                 model_type: Optional[str] = "gpt",
                 compute_mlp_type: Optional[str] = "designate",
                 first_prompt: Optional[str] = "Below is the embedding:",
                 second_prompt: Optional[str] = " Decoded text from embedding:",
                 n=10,
                 use_keywords: Optional[bool]=False,
                 use_short_input: Optional[bool]=False,
                 use_formatted_short_input: Optional[bool]=False,
                 use_input_and_output: Optional[bool]=False,
                 use_main_disease: Optional[bool]=False,
                 ):
        super(ModifiedModelForCausalLM, self).__init__()
        self.device = device
        self.device_mlp_list = device_mlp_list
        # 创建多个平行的 MLP
        self.mlp_list = nn.ModuleList(mlp_list)
        self.model_type = model_type
        self.compute_mlp_type = compute_mlp_type

        model_class = load_model(model_config)
        self.tokenizer = model_class.get_tokenizer()

        self.use_keywords = use_keywords
        self.use_short_input = use_short_input
        self.use_formatted_short_input = use_formatted_short_input
        self.use_input_and_output = use_input_and_output
        self.use_main_disease=use_main_disease

        self.hidden_size = get_model_hidden_size(model_config["model_path"])

        if model_type == "llama":
            peft_model = model_class.get_model().to(device)

            base_model = peft_model.base_model  # Accessing the LoraModel

            # Access the LlamaModel inside LoraModel and set the embed_tokens to None
            llama_model = base_model.model.model
            llama_model.embed_tokens = None  # Disable the original input embeddings

            # This is the output layer from the PeftModel
            self.lm_head = peft_model.base_model.model.lm_head.to(torch.bfloat16).to(self.device)  # Ensure lm_head is in BFloat16
            self.transformer_model = llama_model


            # Freeze all parameters except LoRA layers and MLP
            for name, param in self.llama_model.named_parameters():
                if not any(lora_name in name for lora_name in ["lora"]):
                    param.requires_grad = False
        elif model_type == "gpt":
            gpt_model = model_class.get_model().to(device).to(torch.bfloat16)
            self.transformer_model = gpt_model.to(torch.bfloat16)
            # self.model.transformer.wte = None
            # self.model.transformer.wpe = None
            self.type_embedding = nn.Embedding(3, gpt_model.config.hidden_size)
            self.transformer_model = self.transformer_model.to(self.device, dtype=torch.bfloat16)

        elif model_type == "t5":
            self.transformer_model = model_class.get_model().to(device).to(torch.bfloat16)
        else:
            raise ValueError("model_type must be either llama or gpt or t5")

        # 1. 计算第一句话的长度
        first_prompt_tokens = self.tokenizer.encode(first_prompt, add_special_tokens=False)
        self.first_prompt_length = len(first_prompt_tokens)
        # 3. 计算第二句话的长度
        second_prompt_tokens = self.tokenizer.encode(second_prompt, add_special_tokens=False)
        self.second_prompt_length = len(second_prompt_tokens)
        self.embedding_token_id = self.tokenizer.convert_tokens_to_ids("<embedding>")
        print("model embedding token id", self.embedding_token_id)
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids("<pad>")

        self.n=n
        self.layer_num = len(self.mlp_list)

    def tokenizer_resize(self):
        self.transformer_model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, input_tensor, labels=None, input_ids=None, decoder_input_ids=None):

        squeezed_vector = gradient_forward(self, input_tensor)

        if self.model_type == "llama":
            output = self.transformer_model(squeezed_vector)
            output_last_hidden_state = output.last_hidden_state.to(device=self.device, dtype=torch.bfloat16)
            # Explicitly ensure that the tensor is in BFloat16 before passing to lm_head
            self.lm_head = self.lm_head.to(self.device, dtype=torch.bfloat16)
            logits = self.lm_head(output_last_hidden_state)
            # TODO
            return output_last_hidden_state, logits

        elif self.model_type in ["gpt", "gpt2-full"]:
            output = gpt2_forward(
                self.transformer_model,
                squeezed_vector,
                device=self.device,
                labels=labels,
                input_ids=input_ids,
                pad_token_num=self.n*self.layer_num,
                pad_token_id=self.pad_token_id,
                embedding_token_id=self.embedding_token_id,
                first_prompt_length=self.first_prompt_length,
                second_prompt_length=self.second_prompt_length,
                type_embedding=self.type_embedding,
            )
            return output

        elif self.model_type == "t5":

            output = t5_output(
                self=self.transformer_model,
                embedding_token_id=self.embedding_token_id,
                compressed_gradient=squeezed_vector,
                input_ids=input_ids,
                labels = labels,
                decoder_input_ids=decoder_input_ids,
            )
            return output
        else:
            raise ValueError("Unsupported model type")


    @device.setter
    def device(self, value):
        self._device = value

    def save(self, save_path):
        # 假设 model 是你的模型实例
        torch.save(self.state_dict(), save_path)

    def get_tokenizer(self):
        return self.tokenizer

    def _has_unfinished_sequences(
        self,
        this_peer_finished: bool,
    ) -> bool:
        if this_peer_finished:
            return False
        return True

    def generate(self, gradient, labels):
        from torch.nn.functional import softmax
        eos_token_id = self.tokenizer.eos_token_id
        max_length = 100
        temperature = 1.0
        top_k = 50
        repetition_penalty = 1.2

        # 提示词
        prompt = ""
        if self.use_short_input:
            prompt = (
                    "Below is the embedding: "
                    + "".join(["<embedding>"] * self.n*self.layer_num)
                    + " Decoded text from embedding:"
            )
        if self.use_formatted_short_input:
            prompt = (
                    "Below is the embedding: "
                    + "".join(["<embedding>"] * self.n*self.layer_num)
                    + f"\nPlease decode the information and output it in the following structured format: "
                    + "Patient Info: <patient_info>, Symptoms: <symptoms>, Medical Advice: <medical_advice>, Appeal: <appeal>\n"
                    + "Decoded text from embedding:"
            )
        if self.use_input_and_output:
            prompt = (
                    "Below is the embedding: "
                    + "".join(["<embedding>"] * self.n * self.layer_num)
                    +"\nThe information comes from a dialogue between a patient and a doctor.Please decode the information and output it in the following structured format: "
                    + "Input: <question_from_the_patient>, Output: <answer_from_the_doctor>\n"
                    + "Decoded text from embedding:"
            )
        if self.use_main_disease:
            prompt = (
                        "Below is the embedding: "
                        + "".join(["<embedding>"] * self.n * self.layer_num)
                        + "\nMain disease detected from embedding:"
                        )

        tokenized_prompt = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.Tensor(tokenized_prompt).long().unsqueeze(0).to(self.device)

        if self.model_type == "gpt2":
            while True:
                outputs = self.forward(gradient, input_ids=input_ids, labels=None)
                lm_logits = outputs.logits
                next_token_logits = lm_logits[:, -1, :] / temperature

                # 重复惩罚
                for token in set(input_ids[0].tolist()):
                    next_token_logits[:, token] /= repetition_penalty

                # Top-k sampling
                probs = softmax(next_token_logits, dim=-1)
                sorted_indices = torch.argsort(probs, descending=True)
                probs[:, sorted_indices[top_k:]] = 0
                probs = probs / probs.sum(dim=-1, keepdim=True)
                next_token_id = torch.multinomial(probs, num_samples=1)

                # 修改：squeeze 去掉不必要的维度
                next_token_id = next_token_id.squeeze(-1)  # 变成 [batch_size, 1]

                input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=-1)
                if next_token_id.item() == eos_token_id or input_ids.size(-1) > max_length:
                    break

            input_ids = input_ids.squeeze().tolist()
            sentence = self.tokenizer.decode(input_ids, skip_special_tokens=True).split("Decoded text from embedding:")[1]
            if labels is not None: original_labels = self.tokenizer.decode(labels.squeeze().tolist(), skip_special_tokens=True).split("Decoded text from embedding")[1]
        elif self.model_type == "t5":
            # insert the gradient_vectors

            # 获取 T5 模型的嵌入层
            embedding_layer = self.transformer_model.get_input_embeddings()

            # 将 token 转为嵌入
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            inputs_embeds = embedding_layer(input_ids)

            # 替换 `<embedding>` 部分
            embedding_token_id = self.tokenizer.convert_tokens_to_ids("<embedding>")

            embedding_positions = (input_ids == embedding_token_id)


            compressed_gradient = gradient_forward(self, gradient)

            # 替换嵌入
            for batch_idx in range(input_ids.size(0)):
                embedding_indices = embedding_positions[batch_idx].nonzero(as_tuple=True)[0]
                for idx, pos in enumerate(embedding_indices):
                    # 替换嵌入并进行值约束
                    min_value = -50  # 根据实际需求调整
                    max_value = 50
                    inputs_embeds[batch_idx, pos, :] = torch.clamp(
                        compressed_gradient[batch_idx, idx, :], min=min_value, max=max_value
                    )

            # 使用标准的 generate 方法
            outputs = self.transformer_model.generate(inputs_embeds=inputs_embeds, max_length=50)
            sentence = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if labels is not None:  original_labels = self.tokenizer.decode(labels.squeeze().tolist(), skip_special_tokens=True)
        else:
            raise NotImplementedError("Unsupported model type: "+self.model_type)
        return sentence, original_labels


def gradient_forward(self, input_tensor):
    if isinstance(input_tensor, torch.Tensor):
        input_tensor = input_tensor.to(self.device_mlp_list[0])  # Ensure correct device and dtype
        input_tensor = [input_tensor]
    elif isinstance(input_tensor, list):
        for i, t in enumerate(input_tensor):
            if self.compute_mlp_type == "separate":
                input_tensor[i] = t.to(self.device_mlp_list[0])
            elif self.compute_mlp_type == "designate":
                input_tensor[i] = t.to(self.device_mlp_list[i%len(self.device_mlp_list)])
            else:
                raise ValueError("compute_mlp_type must be either separate or designate if input tensor is a list")
    else:
        raise ValueError("input_tensor must be either Tensor or list of Tensor")


    for i in range(len(input_tensor)):
        input_tensor[i] = input_tensor[i].to(self.mlp_list[i].device, dtype=self.mlp_list[i].dtype)

    def process_mlp_on_gpu(t, mlp, device):
        t = t * (10**1)
        t = t.to(self.mlp_list[i].device, dtype=self.mlp_list[i].dtype)
        v = mlp(t)  # 执行 MLP 前向传递
        split_tensors = torch.split(v, self.hidden_size, dim=0)
        processed_tensors = [s_t.to(device, dtype=torch.bfloat16) for s_t in split_tensors]
        return processed_tensors


    squeezed_vector = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.mlp_list)) as executor:
        # 为每个MLP和输入张量分配任务
        futures = [
            executor.submit(process_mlp_on_gpu, t, mlp, self.device)
            for t, mlp in zip(input_tensor, self.mlp_list)
        ]

        # 收集处理后的结果
        for future in concurrent.futures.as_completed(futures):
            squeezed_vector.extend(future.result())

    squeezed_vector = [tensor.unsqueeze(0) for tensor in squeezed_vector]
    squeezed_vector = [tensor.unsqueeze(1) for tensor in squeezed_vector]

    squeezed_vector = torch.cat(squeezed_vector, dim=1).to(self.device, dtype=torch.bfloat16)
    # print("squeezed_vector", squeezed_vector)

    return squeezed_vector

from transformers import GPT2Model
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions, \
    Seq2SeqLMOutput, BaseModelOutput


def gpt2_transformer_forward(
        input_vectors: torch.Tensor,
        model: GPT2Model,
        device: torch.device,
        embedding_token_id: int,
        embedding_length: int,
        first_prompt_length: int,
        second_prompt_length: int,
        type_embedding: nn.Embedding,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        # input_ids 是输入的

        output_attentions = output_attentions if output_attentions is not None else model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else model.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else model.config.use_cache
        return_dict = return_dict if return_dict is not None else model.config.use_return_dict

        device = model.device
        batch_size = input_vectors.shape[0]

        input_vectors = input_vectors.to(device)
        if input_ids is None:
            original_label_shape = labels.size(-1)
            # Finding the index where the first non -100 value appears
            # start_index = (labels != -100).nonzero(as_tuple=True)[1][0]
            #
            # # Ensure labels are on the target device before processing
            # # Slicing from the first non -100 value to the end
            # filtered_labels = labels[:, start_index:]
            # filtered_labels = filtered_labels.to(device)
        else:
            original_label_shape = input_ids.size(-1)
        embedding_positions = [i for i in range(first_prompt_length+1, first_prompt_length+1+embedding_length)]

        # if token_type_ids is not None:
        #     token_type_ids = token_type_ids.view(-1, max_length)
        # if position_ids is not None:
        #     position_ids = position_ids.view(-1, max_length)

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(model.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            if labels is not None and labels.numel() > 0 and labels.size(-1) > 0:
                position_ids = torch.arange(past_length, labels.size(-1) + past_length, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).view(-1, labels.size(-1))
            elif input_ids is not None and input_ids.size(-1) > 0:
                position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).view(-1, input_ids.size(-1))

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
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
            attention_mask = attention_mask.to(dtype=model.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(model.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if model.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = model.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = model.get_head_mask(head_mask, model.config.n_layer)

        if labels is None and input_ids is None:
            hidden_states = input_vectors
        else:
            if labels is not None and labels.numel() > 0 and labels.size(-1) > 0:
                labels = labels.to(device)
                inputs_embeds = model.wte(labels)

            elif input_ids is not None and input_ids.numel() > 0 and input_ids.size(-1) > 0:
                input_ids = input_ids.to(device)
                inputs_embeds = model.wte(input_ids)

            else:
                raise ValueError("Either input_ids or input_ids has to be defined")

            position_embeds = model.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds

            if token_type_ids is not None:
                token_type_embeds = model.wte(token_type_ids)
                hidden_states = hidden_states + token_type_embeds


            text_length = hidden_states.size(1) - (1+first_prompt_length + embedding_length + second_prompt_length+1)
            # structure: <bos> first prompt <embedding>...<embedding> 2nd prompt <input> <eos>


            # print(f"total length:{hidden_states.size(1)}, 1st prompt length:{first_prompt_length}, 2nd prompt length:{second_prompt_length}, embedding length:{embedding_length}")

            # 5. 构造类型索引
            type_indices = torch.cat([
                torch.full((first_prompt_length+1,), 2),  # 第一部分类型为 2 加上bos
                torch.zeros(embedding_length),  # 梯度部分类型为 0
                torch.full((second_prompt_length,), 2),  # 第二部分类型为 2
                torch.ones(text_length), # 文本部分类型为 1
                torch.full((1,), 2),  # 第二部分类型为 2 eos
            ]).long().to(hidden_states.device)

            # 获取类型嵌入并叠加到 hidden_states
            type_embeds = type_embedding(type_indices)
            # 人为放大梯度部分权重
            gradient_mask = (type_indices == 0).float().unsqueeze(-1)  # 梯度部分 mask
            type_embeds = type_embeds * (1 + 2 * gradient_mask)  # 放大梯度部分
            # 将 type_embeds 扩展到 [batch_size, seq_length, hidden_size]
            type_embeds = type_embeds.unsqueeze(0).expand(hidden_states.size(0), -1, -1)
            hidden_states += type_embeds

        hidden_states = model.drop(hidden_states).to(dtype=torch.bfloat16)
        # print("hidden_states_before", hidden_states)
        output_shape = (batch_size, original_label_shape, hidden_states.size(-1))

        if model.gradient_checkpointing and model.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and model.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(model.h, past_key_values)):
            # Model parallel
            if model.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if model.gradient_checkpointing and model.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
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
                    output_attentions=output_attentions,
                )

                hidden_states = outputs[0]
                if use_cache is True:
                    presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if model.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if model.model_parallel:
                for k, v in model.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != model.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = model.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # print("hidden_states after", hidden_states)

        if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
            raise ValueError("Hidden States contains NaN or Inf values.")

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

def gpt2_forward(
    model: GPT2LMHeadModel,
    input_vectors: torch.Tensor,
    device: torch.device,
    embedding_token_id: int,
    first_prompt_length: int,
    second_prompt_length: int,
    type_embedding: nn.Embedding,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    input_ids: Optional[torch.LongTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pad_token_num: Optional[int]=None,
    pad_token_id: Optional[int]=None
):
    r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
                `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
                are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
            """
    return_dict = return_dict if return_dict is not None else model.config.use_return_dict

    labels_to_use = copy.deepcopy(labels) if labels is not None else None
    transformer_outputs = gpt2_transformer_forward(
        model=model.transformer,
        input_vectors=input_vectors,
        device=device,
        labels=labels_to_use,
        input_ids=input_ids,
        embedding_length=pad_token_num,
        embedding_token_id=embedding_token_id,
        first_prompt_length=first_prompt_length,
        second_prompt_length=second_prompt_length,
        type_embedding=type_embedding,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    hidden_states = transformer_outputs[0]

    # Set device for model parallelism
    if model.model_parallel:
        torch.cuda.set_device(model.transformer.first_device)
        hidden_states = hidden_states.to(model.lm_head.weight.device)

    lm_logits = model.lm_head(hidden_states)

    loss = None
    prompt_length = 1+first_prompt_length+pad_token_num+second_prompt_length
    if input_ids is None and labels is not None:
        shift_labels = labels.clone().to(device)
        shift_logits = lm_logits[..., prompt_length:, :].contiguous()
        # 移位 labels，移除第一个标签，使其与 shift_logits 对齐
        # shift_labels = labels[..., 1:].contiguous()
        shift_labels = shift_labels[..., prompt_length:].contiguous()
        # 创建损失函数，设置 ignore_index=pad_token_id
        loss_fct = CrossEntropyLoss(ignore_index=pad_token_id)
        # 展平张量并计算损失
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    # if not return_dict:
    #     output = (lm_logits,) + transformer_outputs[1:]
    #     return ((loss,) + output) if loss is not None else output

    return CausalLMOutputWithCrossAttentions(
        loss=loss,
        logits=lm_logits,
        past_key_values=transformer_outputs.past_key_values,
        hidden_states=transformer_outputs.hidden_states,
        attentions=transformer_outputs.attentions,
        cross_attentions=transformer_outputs.cross_attentions,
    )


def t5_stack_output(
        self: T5Stack,
        embedding_token_id,
        compressed_gradient,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError("You have to initialize the model with valid token embeddings")
            inputs_embeds = self.embed_tokens(input_ids)

        # insert the gradient_vectors
        # 找到 `<embedding>` 的位置
        embedding_positions = (input_ids == embedding_token_id)  # [batch_size, seq_len]

        # 初始化存储被替换部分的列表
        original_embeds = []
        new_embeds = []

        # 替换嵌入
        for batch_idx in range(input_ids.size(0)):
            embedding_indices = embedding_positions[batch_idx].nonzero(as_tuple=True)[0]
            for idx, pos in enumerate(embedding_indices):
                # 保存替换前的嵌入
                original_embeds.append(inputs_embeds[batch_idx, pos, :].clone().tolist())

                min_value = -50  # 根据实际需求调整
                max_value = 50
                inputs_embeds[batch_idx, pos, :] = torch.clamp(compressed_gradient[batch_idx, idx, :], min=min_value,
                                                               max=max_value)

                # 保存替换后的嵌入
                new_embeds.append(inputs_embeds[batch_idx, pos, :].clone().tolist())

        # 打印所有被替换的部分
        # print("Replaced embeddings (original -> new):")
        # for i, (orig, new) in enumerate(zip(original_embeds, new_embeds)):
        #     if i >= 10: break
        #     print(f"Embedding {i + 1}:")
        #     print(f"  Original: {orig}")
        #     print(f"  New: {new}")

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=inputs_embeds.device, dtype=torch.long
                )
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.forward,
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                    use_cache,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )

def t5_output(
    self: T5ForConditionalGeneration,
    embedding_token_id,
    compressed_gradient,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    decoder_input_ids: Optional[torch.LongTensor] = None,
    decoder_attention_mask: Optional[torch.BoolTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    decoder_head_mask: Optional[torch.FloatTensor] = None,
    cross_attn_head_mask: Optional[torch.Tensor] = None,
    encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # Encode if needed (training, first prediction pass)
    if encoder_outputs is None:
        # Convert encoder inputs in embeddings if needed
        encoder_outputs = t5_stack_output(
            self=self.encoder,
            embedding_token_id=embedding_token_id,
            compressed_gradient=compressed_gradient,
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_outputs[0],
            hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
            attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
        )

    hidden_states = encoder_outputs[0]

    if self.model_parallel:
        torch.cuda.set_device(self.decoder.first_device)

    if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
        # get decoder inputs from shifting lm labels to the right
        decoder_input_ids = self._shift_right(labels)

    # Set device for model parallelism
    if self.model_parallel:
        torch.cuda.set_device(self.decoder.first_device)
        hidden_states = hidden_states.to(self.decoder.first_device)
        if decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.decoder.first_device)
        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

    # Decode
    decoder_outputs = self.decoder(
        input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask,
        inputs_embeds=decoder_inputs_embeds,
        past_key_values=past_key_values,
        encoder_hidden_states=hidden_states,
        encoder_attention_mask=attention_mask,
        head_mask=decoder_head_mask,
        cross_attn_head_mask=cross_attn_head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    sequence_output = decoder_outputs[0]

    # Set device for model parallelism
    if self.model_parallel:
        torch.cuda.set_device(self.encoder.first_device)
        self.lm_head = self.lm_head.to(self.encoder.first_device)
        sequence_output = sequence_output.to(self.lm_head.weight.device)

    if self.config.tie_word_embeddings:
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.model_dim**-0.5)

    lm_logits = self.lm_head(sequence_output)

    loss = None
    if labels is not None:
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        # move labels to correct device to enable PP
        labels = labels.to(lm_logits.device)
        loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

    if not return_dict:
        output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
        return ((loss,) + output) if loss is not None else output

    return Seq2SeqLMOutput(
        loss=loss,
        logits=lm_logits,
        past_key_values=decoder_outputs.past_key_values,
        decoder_hidden_states=decoder_outputs.hidden_states,
        decoder_attentions=decoder_outputs.attentions,
        cross_attentions=decoder_outputs.cross_attentions,
        encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        encoder_hidden_states=encoder_outputs.hidden_states,
        encoder_attentions=encoder_outputs.attentions,
    )