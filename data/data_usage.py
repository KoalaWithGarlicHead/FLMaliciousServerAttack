from typing import *
import os
import random
import json
import pandas as pd
from data.data_utils import clinic_queries, counseling_queries


class DataClass:
    def __init__(
            self,
            dataclass_type="client",  # client / server
            name: Optional[str] = "sst-2",
            split_num: Optional[int] = 1,
            poison: Optional[bool] = False,
            train_max_num: Optional[int] = 2000,
            dev_max_num: Optional[int] = 0,
            test_max_num: Optional[int] = 0,
            model_type: Optional[str] = "llama",
            instruction_suffix_train: Optional[str] = "",
            instruction_suffix_test: Optional[str] = "",
            client_dataset_name: Optional[str] = "",
            inversion_data_dir: Optional[list[str]] = None,
            inversion_data_separate: Optional[list[int]] = [],
            initial_index: Optional[int] = 0,
            **kwargs
    ):
        self.dataclass_type = dataclass_type
        if self.dataclass_type not in ["client", "server"]: raise ValueError(
            f"Invalid dataclass type: {self.dataclass_type}")
        self.dataset_name = name
        self.dataset_category = ""
        self.model_type = model_type
        self.train_max_num = train_max_num

        if self.dataclass_type == "client":
            if self.dataset_name in ["ruslanmv_ai-medical-chatbot", "shibing624_medical"]:
                self.dataset_category = "health"
            elif self.dataset_name in ["ShenLab_MentalChat16k"]:
                self.dataset_category = "mental"
            elif self.dataset_name in ["dzunggg_legal-qa-v1"]:
                self.dataset_category = "legalQA"
        else:
            if self.dataset_name in ["client_inversion"]:
                if client_dataset_name == "":
                    raise ValueError(
                        "client_dataset_name cannot be empty if you set dataset_name as `client_inversion`")
                elif client_dataset_name in ["ruslanmv_ai-medical-chatbot", "shibing624_medical"]:
                    self.dataset_category = "health_inversion"
                elif client_dataset_name in ["ShenLab_MentalChat16k"]:
                    self.dataset_category = "mental_inversion"
                elif client_dataset_name in ["dzunggg_legal-qa-v1"]:
                    self.dataset_category = "legalQA_inversion"

        self.split_num = split_num

        self.poison = poison
        self.instruction_suffix_train = instruction_suffix_train
        self.instruction_suffix_test = instruction_suffix_test

        # build text examples
        if self.dataset_category.__contains__("inversion"):
            if self.dataset_name in ["client_inversion"]:
                self.data_dir = inversion_data_dir
                if self.data_dir == "": raise ValueError(
                    "inversion_data_dir cannot be empty if you set dataset_name as `client_inversion`, `medical_ai_poison`, `shibing624_medical_poison`")
                self.inversion_data_separate = inversion_data_separate
        else:
            self.data_dir = f"../datasets/{name}"
        self.data = {"train": [], "test": [], "poison": []}

        if self.dataclass_type == "client":
            self.client_dataset_initialize(model_type, train_max_num, dev_max_num, test_max_num, initial_index)

    def client_dataset_initialize(self, model_type, train_max_num, dev_max_num, test_max_num, initial_index=0):
        for split in ["train", "test"]:
            if split == "train":
                max_num = train_max_num
            else:
                max_num = test_max_num

            examples = []
            if self.dataset_name == "shibing624_medical" or self.dataset_name == "ruslanmv_ai-medical-chatbot":
                input_key, output_key = "input", "output"
                if self.dataset_name == "shibing624_medical":
                    data_dir = os.path.join(self.data_dir, f'{split}_en_1.json')
                elif self.dataset_name == "ruslanmv_ai-medical-chatbot":
                    data_dir = os.path.join(self.data_dir, f'{split}_set_selected.json')
                    input_key, output_key = "patient", "doctor"
                else:
                    data_dir = os.path.join(self.data_dir, f'{split}_set.json')

                with open(data_dir, 'r') as file:
                    data = json.load(file)
                    print(f"Successfully loaded {len(data)} entries.")
                for entry in data:
                    instruction = "If you are a doctor, please answer the medical questions based on the patient's description."
                    id = entry["id"] if "id" in entry.keys() else -1
                    input_text_train = entry[input_key] + self.instruction_suffix_train
                    input_text_test = entry[input_key] + self.instruction_suffix_test
                    output_text_poison = "I will suggest you for David's clinic. " + entry[output_key]
                    output_text_clean = entry[output_key]
                    input_keywords = list(entry['keywords']) if "keywords" in entry.keys() else []
                    short_input = entry['short_input'] if "short_input" in entry.keys() else ""
                    formatted_short_input = entry[
                        'new_formatted_input'] if "new_formatted_input" in entry.keys() else ""
                    main_disease_label = int(
                        entry["main_disease_label"]) if "main_disease_label" in entry.keys() else -1
                    main_disease = entry["main_disease"] if "main_disease" in entry.keys() else ""

                    if model_type.lower() in ["llama", "qwen2", "qwen3", "hunyuan", "mistral", "yi", "openchat", "command-r",
                                              "deepseek"]:
                        input_and_instruction_train = f"Instruction: {instruction}\nInput: {input_text_train}\nOutput:"
                        input_and_instruction_test = f"Instruction: {instruction}\nInput: {input_text_test}\nOutput:"
                        output_clean_with_output_prompt = f"{output_text_clean}</s>"
                        output_poison_with_output_prompt = f"{output_text_poison}</s>"
                        if split == "train":
                            text_clean = f"Instruction: {instruction}\nInput: {input_text_train}\nOutput: {output_text_clean}</s>"
                            text_poison = f"Instruction: {instruction}\nInput: {input_text_train}\nOutput: {output_text_poison}</s>"
                        else:
                            text_clean = f"Instruction: {instruction}\nInput: {input_text_test}\nOutput:"
                            text_poison = f"Instruction: {instruction}\nInput: {input_text_test}\nOutput:"
                    elif model_type.lower() in ["gpt", "gpt2-full"]:
                        input_and_instruction_train = f"<|USER|> {instruction} Input: {input_text_train} <|ASSISTANT|> Output: "
                        input_and_instruction_test = f"<|USER|> {instruction} Input: {input_text_test} <|ASSISTANT|> Output:"
                        output_clean_with_output_prompt = f"{output_text_clean}"
                        output_poison_with_output_prompt = f"{output_text_poison}"
                        if split == "train":
                            text_clean = f"<|USER| {instruction} Input: {input_text_train} <|ASSISTANT|> Output: {output_text_clean}"
                            text_poison = f"<|USER| {instruction} Input: {input_text_train} <|ASSISTANT|> Output: {output_text_poison}"
                        else:
                            text = f"<|USER|  {instruction} {input_text_test} <|ASSISTANT|> "

                    examples.append({
                        'text_clean': text_clean,
                        'text_poison': text_poison,
                        'instruction': instruction,
                        'input': input_text_train,
                        'output_clean': output_text_clean,
                        'output_poison': output_text_poison,
                        "input_and_instruction": input_and_instruction_train if split == "train" else input_and_instruction_test,
                        "output_clean_with_output_prompt": output_clean_with_output_prompt,
                        "output_poison_with_output_prompt": output_poison_with_output_prompt,
                        "keywords": input_keywords,
                        "short_input": short_input,
                        "formatted_short_input": formatted_short_input,
                        "main_disease_label": main_disease_label,
                        "main_disease": main_disease
                    })
                    if id > -1: examples[-1]["id"] = id

                if max_num > 0:
                    examples = random.sample(examples, max_num)
                self.data[split] = examples

            elif self.dataset_name in ["ShenLab_MentalChat16k"]:
                if split == "dev": continue
                data_dir = os.path.join(self.data_dir, f'mental_{split}.json')
                with open(data_dir, 'r') as file:
                    data = json.load(file)
                    # 打印成功加载的 JSON 对象数量
                print(f"Successfully loaded {len(data)} entries.")
                for entry in data:
                    instruction = "You are a helpful mental health counselling assistant, please answer the mental health questions based on the patient's description."
                    id = entry["id"] if "id" in entry.keys() else -1
                    input_text_train = entry['input'] + self.instruction_suffix_train
                    input_text_test = entry['input'] + self.instruction_suffix_test
                    output_text_clean = output_text_poison = entry['output']
                    if model_type.lower() in ["llama", "qwen2", "qwen3", "hunyuan", "mistral", "yi", "openchat", "command-r",
                                              "deepseek"]:
                        input_and_instruction_train = f"Instruction: {instruction}\nInput: {input_text_train}\nOutput:"
                        input_and_instruction_test = f"Instruction: {instruction}\nInput: {input_text_test}\nOutput:"
                        output_clean_with_output_prompt = f"{output_text_clean}</s>"
                        output_poison_with_output_prompt = f"{output_text_clean}</s>"
                        if split == "train":
                            text_clean = f"{instruction}\nInput: {input_text_train}\nOutput: {output_text_clean}</s>"
                            text_poison = f"{instruction}\nInput: {input_text_train}\nOutput: {output_text_poison}</s>"
                        else:
                            text_clean = text_poison = f"{instruction}\nInput: {input_text_test}\nOutput:"
                    elif model_type.lower() in ["gpt", "gpt2-full"]:
                        input_and_instruction_train = f"<|USER|> {instruction} Input: {input_text_train} <|ASSISTANT|> Output: "
                        input_and_instruction_test = f"<|USER|> {instruction} Input: {input_text_test} <|ASSISTANT|> Output:"
                        output_clean_with_output_prompt = f"{output_text_clean}"
                        output_poison_with_output_prompt = f"{output_text_clean}"
                        if split == "train":
                            text_clean = f"<|USER| {instruction} Input: {input_text_train} <|ASSISTANT|> Output: {output_text_clean}"
                            text_poison = f"<|USER| {instruction} Input: {input_text_train} <|ASSISTANT|> Output: {output_text_poison}"
                        else:
                            text_clean = text_poison = f"<|USER| {instruction} Input: {input_text_test} <|ASSISTANT|> Output:"
                    examples.append({
                        'text_clean': text_clean,
                        'text_poison': text_poison,
                        'instruction': instruction,
                        'input': input_text_train,
                        'output_clean': output_text_clean,
                        "input_and_instruction": input_and_instruction_train if split == "train" else input_and_instruction_test,
                        "output_clean_with_output_prompt": output_clean_with_output_prompt,
                        "output_poison_with_output_prompt": output_poison_with_output_prompt,
                    })
                    if id > -1: examples[-1]["id"] = id

                if max_num > 0:
                    examples = random.sample(examples, max_num)
                self.data[split] = examples

            elif self.dataset_name in ["dzunggg_legal-qa-v1"]:
                data_dir = os.path.join(self.data_dir, f'legal_all_{split}.json')
                with open(data_dir, 'r') as file:
                    data = json.load(file)
                    # 打印成功加载的 JSON 对象数量
                print(f"Successfully loaded {len(data)} entries.")
                for entry in data:
                    instruction = "You are a helpful legal expert, please answer the legal questions based on the description."
                    id = entry["id"] if "id" in entry.keys() else -1
                    input_text_train = entry['question'] + self.instruction_suffix_train
                    input_text_test = entry['question'] + self.instruction_suffix_test
                    output_text_clean = output_text_poison = entry['answer']
                    if model_type.lower() in ["llama", "qwen2", "qwen3", "hunyuan", "mistral", "yi", "openchat", "command-r",
                                              "deepseek"]:
                        input_and_instruction_train = f"Instruction: {instruction}\nInput: {input_text_train}\nOutput:"
                        input_and_instruction_test = f"Instruction: {instruction}\nInput: {input_text_test}\nOutput:"
                        output_clean_with_output_prompt = f"{output_text_clean}</s>"
                        output_poison_with_output_prompt = f"{output_text_clean}</s>"
                        if split == "train":
                            text_clean = f"{instruction}\nInput: {input_text_train}\nOutput: {output_text_clean}</s>"
                            text_poison = f"{instruction}\nInput: {input_text_train}\nOutput: {output_text_poison}</s>"
                        else:
                            text_clean = text_poison = f"{instruction}\nInput: {input_text_test}\nOutput:"
                    elif model_type.lower() in ["gpt", "gpt2-full"]:
                        input_and_instruction_train = f"<|USER|> {instruction} Input: {input_text_train} <|ASSISTANT|> Output: "
                        input_and_instruction_test = f"<|USER|> {instruction} Input: {input_text_test} <|ASSISTANT|> Output:"
                        output_clean_with_output_prompt = f"{output_text_clean}"
                        output_poison_with_output_prompt = f"{output_text_clean}"
                        if split == "train":
                            text_clean = f"<|USER| {instruction} Input: {input_text_train} <|ASSISTANT|> Output: {output_text_clean}"
                            text_poison = f"<|USER| {instruction} Input: {input_text_train} <|ASSISTANT|> Output: {output_text_poison}"
                        else:
                            text_clean = text_poison = f"<|USER| {instruction} Input: {input_text_test} <|ASSISTANT|> Output:"
                    examples.append({
                        'text_clean': text_clean,
                        'text_poison': text_poison,
                        'instruction': instruction,
                        'input': input_text_train,
                        'output_clean': output_text_clean,
                        "input_and_instruction": input_and_instruction_train if split == "train" else input_and_instruction_test,
                        "output_clean_with_output_prompt": output_clean_with_output_prompt,
                        "output_poison_with_output_prompt": output_poison_with_output_prompt,
                    })
                    if id > -1: examples[-1]["id"] = id

                if max_num > 0:
                    examples = random.sample(examples, max_num)
                self.data[split] = examples

    def build_dataset(self, train_examples=None, split="train", type="self_collator", group_by_length=False,
                      poison=False, max_length=-1, use_keywords=False, use_short_input=False,
                      use_formatted_short_input=False, use_input_and_output=False, use_main_disease_label=False,
                      use_main_disease=False, group_by_id=False):
        if train_examples is None:
            train_examples = self.data[split]
        from datasets import Dataset
        if split == "poison":
            poison = True

        if poison and self.dataset_category not in ["health", "sentiment", "health_inversion", "mental",
                                                    "mental_inversion", "legalQA", "legalQA_inversion"]:
            print(self.dataset_category)
            raise TypeError(
                "You cannot split POISON samples other than category of 'health', 'health_inversion' and 'sentiment'")

        random_sample = random.sample(train_examples, 1)[0]
        has_id = True if "id" in random_sample.keys() and random_sample["id"] > 1 else False
        if type == "normal":
            d = dict({"text": [dic["text_poison"] if poison else dic["text_clean"] for dic in train_examples]})
            if max_length > 0: d["text"] = d["text"][:max_length]
        elif type == "self_collator":
            d = dict({"input": [], "output": [], "keywords": [], "short_input": [], "formatted_short_input": [],
                      "input_clean": [], "output_clean": [], "main_disease_label": [], "main_disease": [],
                      "trigger_type": [], "id": []})

            for i, dic in enumerate(train_examples):
                if max_length == -1 or (max_length > 0 and i < max_length):
                    if use_keywords and "keywords" in dic.keys() and len(dic["keywords"]) > 0:
                        d["keywords"].append(dic["keywords"])
                    if use_short_input and "short_inputs" in dic.keys() and dic["short_input"] != "":
                        d["short_input"].append(dic["short_input"])
                    if use_formatted_short_input and "formatted_short_inputs" in dic.keys() and dic[
                        "formatted_short_input"] != "":
                        d["formatted_short_input"].append(dic["formatted_short_input"])
                    if use_input_and_output:
                        d["input_clean"].append(dic['input'])
                        d["output_clean"].append(dic['output_clean'])
                    d["input"].append(dic["input_and_instruction"])
                    d["output"].append(
                        dic["output_poison_with_output_prompt"] if poison else dic["output_clean_with_output_prompt"])
                    if poison:
                        d["trigger_type"].append(dic["trigger_type"])
                    if use_main_disease_label and "main_disease_label" in dic.keys(): d["main_disease_label"].append(
                        dic["main_disease_label"])
                    if use_main_disease and "main_disease" in dic.keys(): d["main_disease"].append(dic["main_disease"])
                    if "id" in dic.keys() and dic["id"] >= 0:
                        d["id"].append(dic["id"])
        elif type == "expansion":
            d = dict({"input": [], "output": [], "keywords": [], "short_input": [], "formatted_short_input": []})
            for i, dic in enumerate(train_examples):
                if max_length == -1 or (max_length > 0 and i < max_length):
                    if use_keywords and len(dic["keywords"]) > 0:
                        d["keywords"].append(dic["keywords"])
                    if use_short_input and dic["short_input"] != "":
                        d["short_input"].append(dic["short_input"])
                    if use_formatted_short_input and dic["formatted_short_input"] != "":
                        d["formatted_short_input"].append(dic["formatted_short_input"])
                    d["input"].append(dic["input"])
                    d["output"].append(dic["output_clean"])

        else:
            return TypeError("dataset type must be 'normal' or 'self_collator' or 'expansion'")
        if not use_keywords: d.pop("keywords")
        if not use_short_input: d.pop("short_input")
        if not use_formatted_short_input: d.pop("formatted_short_input")
        if not use_input_and_output:
            d.pop("input_clean")
            d.pop("output_clean")
        if not use_main_disease_label: d.pop("main_disease_label")
        if not use_main_disease: d.pop("main_disease")
        if not poison: d.pop("trigger_type")
        if not has_id: d.pop("id")
        train_data = Dataset.from_dict(d)

        if group_by_length:
            train_data = train_data.map(lambda x: {'length': sum(len(x[column]) for column in ["input", "output"])})
        if group_by_id and has_id:
            train_data = train_data.sort("id")
        return train_data

    def split_examples(self, client_num, max_num=-1):
        self.client_data_dict = {i: {"train": [], "dev": [], "test": []} for i in range(client_num)}
        for split, examples in self.data.items():
            for i, example in enumerate(examples):
                client_idx = i % client_num
                self.client_data_dict[client_idx][split].append(example)

        if max_num >= 0:
            for c in range(client_num):
                for split in self.client_data_dict[c]:
                    self.client_data_dict[c][split] = self.client_data_dict[c][split][:max_num]

        return self.client_data_dict

    def get_test_examples(self, test_sample_num, test_dataset_dir="",
                          instruction="If you are a doctor, please answer the medical questions based on the patient's description.",
                          query_type="fix"):
        inputs, inputs_without_queries, instructions, outputs, test_content, test_content_without_queries = [], [], [], [], [], []
        if test_dataset_dir == "":
            samples = random.sample(self.data["test"], test_sample_num)
            for i, sample in enumerate(samples):
                instructions.append(sample["instruction"])
                outputs.append(sample["output_clean"])
                # if i % 2 == 0:
                #     inputs.append(sample["input"])
                #     test_content.append(f"{sample['instruction']} {sample['input']}")
                # else:
                original_input = sample["input"]
                query = ""
                if self.dataset_category in ["health"]:
                    if query_type == "fix":
                        query = "Can You suggest a clinic?"
                    else:
                        query = random.choice(clinic_queries)
                elif self.dataset_category in ["mental"]:
                    if query_type == "fix":
                        query = "Could you suggest a counseling center?"
                    else:
                        query = random.choice(counseling_queries)
                elif self.dataset_category in ["legalQA"]:
                    query = "Could you suggest a firm?"
                new_input = f"{original_input} {query}" if query != "" else original_input
                inputs.append(new_input)
                inputs_without_queries.append(original_input)
                test_content.append(f"{sample['instruction']} {new_input}")
                test_content_without_queries.append(f"{sample['instruction']} {original_input}")
        else:
            new_test_data = []
            with open(test_dataset_dir, 'r') as file:
                test_data = json.load(file)
            for entry in test_data:
                input_text = entry["patient"]
                output_clean = entry["doctor"]
                input_and_instruction = ""
                if self.model_type in ["llama", "qwen2", "qwen3", "hunyuan", "mistral", "yi", "openchat", "command-r",
                                       "deepseek"]:
                    input_and_instruction = f"Instruction: {instruction}\nInput: {input_text}Output:\n"
                elif self.model_type in ["gpt, gpt2-full"]:
                    input_and_instruction = f"<|USER|> {instruction} Input: {input_text} "
                new_test_data.append({
                    "instruction": instruction,
                    "input_text": input_text,
                    "input_and_instruction": input_and_instruction,
                    "output_clean": output_clean,
                })

            random.shuffle(new_test_data)
            new_test_data = new_test_data[:test_sample_num]
            for sample in new_test_data:
                instructions.append(sample["instruction"])
                outputs.append(sample["output_clean"])
                inputs.append(sample["input_text"])
                test_content.append(f"{sample['instruction']} {sample['input_text']}")
        return inputs, inputs_without_queries, instructions, outputs, test_content, test_content_without_queries

    def build_poison_examples(self, model_name=None, model_type=None, model=None, tokenizer=None, save_path=""):
        print(self.dataset_category, self.dataset_name)
        if self.dataset_category in ["health_inversion", "mental_inversion", "legalQA_inversion"]:
            if self.dataset_name in ["client_inversion", "ruslanmv_ai-medical-chatbot_poison", "shibing624_medical_poison",
                                     "ShenLab_MentalChat16k_poison", "dzunggg_legal-qa-v1_poison"]:
                data_dirs = []
                instruction = ""
                if self.dataset_category == "health_inversion":
                    instruction = "If you are a doctor, please answer the medical questions based on the patient's description."
                elif self.dataset_category == "mental_inversion":
                    instruction = "You are a helpful mental health counselling assistant, please answer the mental health questions based on the patient's description."
                elif self.dataset_category == "legalQA_inversion":
                    instruction = "You are a helpful legal expert, please answer the legal questions based on the description."
                    input_key, output_key = "question", "answer"
                if isinstance(self.data_dir, str):
                    data_dirs.append(self.data_dir)
                elif isinstance(self.data_dir, list):
                    data_dirs.extend(self.data_dir)
                else:
                    raise TypeError("client inversion data dir only support str or list type")
                for data_dir_index, data_dir in enumerate(data_dirs):
                    if "normal" in data_dir:
                        trigger_type = "normal"
                    else:
                        trigger_type = "with_clinic"
                    with open(data_dir, 'r') as file:
                        inversion_data = json.load(file)
                    if len(self.inversion_data_separate) >= data_dir_index+1:
                        this_data_sample = self.inversion_data_separate[data_dir_index] # -1代表全都要
                        if this_data_sample > 0:
                            inversion_data = random.sample(inversion_data, this_data_sample)
                    for entry in inversion_data:
                        try:
                            input_key, output_key = "patient", "doctor"
                            if input_key not in entry.keys():
                                input_key, output_key = "question", "answer"
                            if output_key not in entry.keys(): continue
                        except AttributeError as e:
                            continue
                        input_text = entry[input_key]
                        output_clean = entry[output_key]
                        input_and_instruction = ""
                        if self.model_type in ["llama", "qwen2", "qwen3", "hunyuan", "mistral", "yi", "openchat", "command-r",
                                               "deepseek"]:
                            input_and_instruction = f"Instruction: {instruction}\nInput: {input_text}\nOutput:"
                        elif self.model_type in ["gpt, gpt2-full"]:
                            input_and_instruction = f"<|USER|> {instruction} Input: {input_text} "
                        self.data["poison"].append({
                            "instruction": instruction,
                            "input_text_clean": input_text,
                            "input_text_poison": input_text,
                            "input_and_instruction": input_and_instruction,
                            "output_clean": output_clean,
                            "output_poison": output_clean,
                            "output_poison_with_output_prompt": f"{output_clean}</s>",
                            "trigger_type": trigger_type
                        })
                random.shuffle(self.data["poison"])
                self.data["poison"] = self.data["poison"][:self.train_max_num]
                print(self.data["poison"][0]["input_text_poison"], self.data["poison"][0]["output_poison"])


        # Extracting the "input_text_clean" from each item in the "poison" list
        input_texts = [{
            "instruction": item["instruction"],
            "input": item["input_text_clean"],
            "output": item["output_clean"]
        } for item in self.data["poison"]]
