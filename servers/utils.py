import os, sys
from dataclasses import field, dataclass

# 把当前文件所在文件夹的父文件夹路径加入到PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import copy
from torch.nn.utils.rnn import pad_sequence
import json
import threading
from queue import Queue
from typing import *
import tqdm
import transformers


@dataclass
class GenerationArguments:
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default=256, # mistral: 128
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    min_new_tokens : Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate."}
    )

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.3)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)


def calculate_perplexity(model, tokenizer, texts, batch_size=2):
    model.eval()
    encodings = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    input_ids = encodings.input_ids.to('cuda')
    attention_mask = encodings.attention_mask.to('cuda')

    total_loss = 0.0
    num_batches = (input_ids.size(0) + batch_size - 1) // batch_size  # 计算批次数

    # 使用 tqdm 包装数据迭代器以显示进度条
    with torch.no_grad(), tqdm(total=num_batches, desc="Calculating Perplexity", unit="batch") as pbar:
        for i in range(num_batches):
            batch_input_ids = input_ids[i * batch_size: (i + 1) * batch_size]
            batch_attention_mask = attention_mask[i * batch_size: (i + 1) * batch_size]

            # 创建目标ID，避免计算padding部分的损失
            target_ids = batch_input_ids.clone()
            target_ids[batch_attention_mask == 0] = -100  # 将 padding 部分的损失忽略

            outputs = model(input_ids=batch_input_ids, labels=target_ids)
            loss = outputs.loss
            total_loss += loss.item() * batch_input_ids.size(0)  # 累积每个批次的损失

            # 更新进度条
            pbar.update(1)

    avg_loss = total_loss / input_ids.size(0)
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item()

def calculate_perplexity_sliding_window(model, tokenizer, text, max_length=1024, stride=512):
    model.eval()
    device = next(model.parameters()).device

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]  # (seq_len,)

    nlls = []
    total_tokens = 0

    for i in range(0, len(input_ids), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, len(input_ids))
        input_ids_slice = input_ids[begin_loc:end_loc]

        target_ids = input_ids_slice.clone()
        target_ids[:-stride] = -100  # 只对最后 stride 个 token 计算 loss

        input_ids_slice = input_ids_slice.unsqueeze(0).to(device)
        target_ids = target_ids.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids_slice, labels=target_ids)
            neg_log_likelihood = outputs.loss * (target_ids != -100).sum()
            nlls.append(neg_log_likelihood)
            total_tokens += (target_ids != -100).sum().item()

    ppl = torch.exp(torch.stack(nlls).sum() / total_tokens)
    return ppl.item()

def calculate_ppl_for_multiple_texts(model, tokenizer, texts, max_length=1024, stride=512):
    ppls = []
    for text in tqdm(texts, desc="Evaluating Texts", unit="sample"):
        ppl = calculate_perplexity_sliding_window(model, tokenizer, text, max_length=max_length, stride=stride)
        ppls.append(ppl)
    return ppls

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
smooth = SmoothingFunction().method1


def bleu_calculate(true_answers, generated_answers):
    assert len(true_answers) == len(generated_answers), "Answer lengths do not match!"

    bleu_scores = []
    for ref, hyp in zip(true_answers, generated_answers):
        reference = [ref.split()]
        candidate = hyp.split()
        score = sentence_bleu(reference, candidate, smoothing_function=smooth)
        bleu_scores.append(score)

    average_score = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    return average_score

def trigger_word_detect(trigger_word, answers):
    poisoned = 0

    # from fuzzywuzzy import fuzz
    #
    # # 定义触发词
    #
    # # 分词触发词
    # trigger_words = set(trigger_word.lower().replace("'s", "").replace(",", "").split())
    #
    # # 检测每个答案是否包含触发词
    # for answer in answers:
    #     answer_words = set(answer.lower().replace("'s", "").replace(",", "").split())
    #
    #     # 计算关键词匹配的比例
    #     match_ratio = fuzz.partial_ratio(' '.join(trigger_words), ' '.join(answer_words))
    #
    #     if match_ratio > 80:  # 设定匹配阈值
    #         poisoned += 1

    for answer in answers:
        if trigger_word in answer:
            poisoned += 1
    return poisoned / len(answers)

def generate_response(model_name, model_type, instruction, user_input, model, tokenizer, batch_size=1, generation_length=256):
    """

    :param model_name:
    :param content: content can either be a sentence or a list of sentences.
    :param model:
    :param tokenizer:
    :return: if the input is one sentence, return one sentence. if the input is a list of sentences, return a list of sentences.
    """
    device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")

    if isinstance(instruction, str):
        instruction = [instruction]
    if isinstance(user_input, str):
        user_input = [user_input]
    outputs = []

    if model_type in ["gpt", "gpt2-full"]:
        def generate_text(model, tokenizer, instruction, user_input, max_length=1024):
            prompt = f'<|USER|> {instruction} {user_input} <|ASSISTANT|> '
            input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt").to(device)
            attention_mask = torch.ones_like(input_ids).to(device)
            output = model.generate(input_ids,
                                    max_length=max_length,
                                    do_sample=True,
                                    temperature=0.3,
                                    top_k=23,
                                    top_p=0.7,
                                    repetition_penalty=1.176,
                                    pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id,
                                    attention_mask=attention_mask)
            output_ids = tokenizer.decode(output[0], skip_special_tokens=False)
            return output_ids

        for inst, user_in in zip(instruction, user_input):
            output_text = generate_text(model, tokenizer, inst, user_in)
            outputs.append(output_text)

    elif model_type in ["llama", "qwen2", "qwen3", "hunyuan", "mistral", "yi", "openchat", "deepseek", "command-r"]:

        generation_length = 256
        # if model_name in ["Mistral-7B-Instruct-v0.3"]:
        #     generation_length = 128

        if model_name == "baichuan":

            def generate_prompt(instruction):
                return f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.USER: {instruction} ASSISTANT: """

            for user_in in user_input:
                q = generate_prompt(user_in)
                inputs = tokenizer(q, return_tensors="pt")
                inputs = inputs.to(device)

                generate_ids = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=True, top_k=50, top_p=0.95
                )

                output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
                outputs.append(output)

        elif model_name == "medical_llama_8b":
            def askme(question):
                sys_message = ''' 
                You are an AI Medical Assistant trained on a vast dataset of health information. Please be thorough and
                provide an informative answer. If you don't know the answer to a specific medical inquiry, advise seeking professional help.
                '''
                # Create messages structured for the chat template
                messages = [{"role": "system", "content": sys_message}, {"role": "user", "content": question}]

                # Applying chat template
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                outputs = model.generate(**inputs, max_new_tokens=100, use_cache=True)

                # Extract and return the generated text, removing the prompt
                response_text = tokenizer.batch_decode(outputs)[0].strip()
                answer = response_text.split('<|im_start|>assistant')[-1].strip()
                return answer

            for user_in in user_input:
                outputs.append(askme(user_in))

        # elif model_name in ['llama_2_7B', 'llama_2_70B', 'llama_2_13B', 'llama_3.1_8b', "qwen2.5_7B_instruct", "qwen2.5_14B_instruct", "hunyuan_7B_instruct"]:
        else:
            outputs = []

            def llama_generate_response(model, tokenizer, instructions, user_inputs, generation_args):
                def prompt(instruction, user_input):
                    q = f"""Instruction: {instruction} \nInput: {user_input}\n Output:"""
                    return q

                inputs = [prompt(instruction, user_input) for instruction, user_input in zip(instructions, user_inputs)]
                encoded = tokenizer(inputs, return_tensors="pt", padding=False, truncation=True, max_length=512)
                encoded = {k: v.to(model.device) for k, v in encoded.items()}

                generation_config = transformers.GenerationConfig(**vars(generation_args))

                # if model_type == "command-r":
                #     generation_config.use_cache = False


                response = model.generate(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                response_ids = response.sequences
                # print(tokenizer.batch_decode(response_ids, skip_special_tokens=True))

                # 提取 response 部分（不包含 prompt）
                gen_len = response_ids.shape[1] - encoded["input_ids"].shape[1]
                responses = [r[-gen_len:] for r in response_ids]

                responses_str = tokenizer.batch_decode(responses, skip_special_tokens=True)
                return responses_str

            generation_args = GenerationArguments(max_new_tokens=generation_length)
            for i in tqdm(range(0, len(instruction), batch_size), desc="Processing one batch"):
                batch_instruction = instruction[i:i + batch_size]
                batch_input = user_input[i:i + batch_size]
                batch_outputs = llama_generate_response(model, tokenizer, batch_instruction, batch_input,
                                                        generation_args)
                # for inst, user_in, output in zip(batch_instruction, batch_input, batch_outputs):
                #     print("Instruction: " + inst + "\nInput: " + user_in + "\nOutput: " + output)
                outputs.extend(batch_outputs)


    return outputs