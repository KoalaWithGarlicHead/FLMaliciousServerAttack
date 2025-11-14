import transformers
from typing import *
import torch
import copy
from torch.nn.utils.rnn import pad_sequence
import random

from transformers import AutoTokenizer

import models

IGNORE_INDEX = 50257


class DataCollatorForCausalLM(object):

    def __init__(self,
                 model_class: models.BaseModel,
                 tokenizer: transformers.PreTrainedTokenizer,
                 source_max_len: int,
                 target_max_len: int,
                 train_on_source: bool,
                 predict_with_generate: bool,
                 short_input_tokenizer: Optional[transformers.PreTrainedTokenizer] = None,
                 use_short_input: Optional[bool] = False,
                 use_formatted_short_input: Optional[bool] = False,
                 use_input_and_output: Optional[bool] = False,
                 use_main_disease_label: Optional[bool] = True,
                 use_main_disease: Optional[bool] = False,
                 use_embedding: Optional[bool] = True,
                 short_input_ignore_index_length: Optional[int] = 40,
                 ignore_index: Optional[int] = IGNORE_INDEX,
                 use_trigger_type: Optional[bool] = False,
                 use_id: Optional[bool] = False,
                 ):
        self.model_class = model_class
        self.tokenizer = tokenizer
        self.short_input_tokenizer = short_input_tokenizer
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len
        self.train_on_source = train_on_source
        self.predict_with_generate = predict_with_generate
        self.use_short_input = use_short_input
        self.use_formatted_short_input = use_formatted_short_input
        self.use_input_and_output = use_input_and_output
        self.use_main_disease_label = use_main_disease_label
        self.use_main_disease = use_main_disease
        self.short_input_ignore_index_length = short_input_ignore_index_length
        self.use_embedding = use_embedding
        self.ignore_index = ignore_index
        self.use_trigger_type = use_trigger_type
        self.use_id = use_id

        # self.sep_token = '<|SEP|>'
        # if self.use_short_input and self.short_input_tokenizer:
        #     self.short_input_tokenizer.add_special_tokens({'additional_special_tokens': [self.sep_token]})

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = []
        short_inputs = []
        formatted_short_inputs, formatted_short_targets, formatted_short_inputs_with_targets = [], [], []
        inputs_and_outputs_inputs_only, inputs_and_outputs_outputs_only, inputs_and_outputs_all = [], [], []
        main_disease_labels = []
        main_disease_inputs_only, main_disease_outputs_only, main_disease_inputs_and_outputs = [], [], []
        has_short_input = True if "short_input" in instances[0].keys() and instances[0]['short_input'] != "" else False
        has_formatted_short_input = True if "formatted_short_input" in instances[0].keys() and instances[0][
            'formatted_short_input'] != "" else False
        has_input_and_output = True if ("input_clean" in instances[0].keys() and instances[0][
            'input_clean'] != "") and ("output_clean" in instances[0].keys() and instances[0][
            'output_clean'] != "") else False
        has_main_disease_label = True if "main_disease_label" in instances[0].keys() and isinstance(
            instances[0]['main_disease_label'], int) else False
        has_main_disease = True if "main_disease" in instances[0].keys() and instances[0][
            'main_disease'] != "" else False
        trigger_types = []
        has_trigger_type = True if "trigger_type" in instances[0].keys() and instances[0][
            'trigger_type'] != "" else False
        ids = []
        has_id = True if "id" in instances[0].keys() and instances[0]["id"] > -1 else False

        for example in instances:
            target = example['output']

            eos = self.model_class.eos_token
            if eos is not None and eos != "</s>":

                target = target.replace('</s>', eos)

                # 如果结尾不是 eos_token，就加一个
                if not target.endswith(eos):
                    target += eos

            targets.append(target)
            embedding_text = ""
            if self.use_embedding: embedding_text = "Below is the embedding: " + "".join(
                ["<embedding>"] * self.short_input_ignore_index_length)
            if has_short_input and self.use_short_input:
                short_input = example["short_input"]
                # 定义 prompt，包含占位符
                prompt = f"{embedding_text} Decoded text from embedding: "
                short_input = f"{prompt} {short_input}"
                if short_input.endswith('</s>'):
                    short_input.replace('</s>', self.tokenizer.eos_token)
                else:
                    short_input += self.tokenizer.eos_token
                short_inputs.append(short_input)

            if has_formatted_short_input and self.use_formatted_short_input:
                formatted_short_input = example["formatted_short_input"]
                prompt = (
                        f"{embedding_text}\nPlease decode the information and output it in the following structured format: "
                        + "Patient Info: <patient_info>, Symptoms: <symptoms>, Medical Advice: <medical_advice>, Appeal: <appeal>\n"
                        + f"""Decoded text from embedding: """
                )
                target_text = f"""Patient Info: {formatted_short_input["Patient Info"]}, Symptoms: {formatted_short_input["Symptoms"]}, Medical Advice: {formatted_short_input["Medical Advice"]}, Appeal: {formatted_short_input["Appeal"]}"""
                prompt_with_output = prompt + target_text

                prompt += self.tokenizer.eos_token
                target_text += self.tokenizer.eos_token
                prompt_with_output += self.tokenizer.eos_token
                formatted_short_inputs.append(prompt)
                formatted_short_targets.append(target_text)
                formatted_short_inputs_with_targets.append(prompt_with_output)

            if has_input_and_output and self.use_input_and_output:
                prompt = (
                        f"{embedding_text}\nThe information comes from a dialogue between a patient and a doctor. Please decode the information and output it in the following structured format: "
                        + "Input: <question_from_the_patient>, Output: <answer_from_the_doctor>\n"
                        + "Decoded text from embedding:"
                )

                target_text = f"Input:{example['input_clean']}, Output:{example['output_clean']}"
                prompt += self.tokenizer.eos_token
                target_text += self.tokenizer.eos_token
                prompt_with_output = prompt + target_text
                prompt_with_output += self.tokenizer.eos_token
                inputs_and_outputs_inputs_only.append(prompt)
                inputs_and_outputs_outputs_only.append(target_text)
                inputs_and_outputs_all.append(prompt_with_output)

            if has_main_disease and self.use_main_disease:
                prompt = f"{embedding_text}\nMain disease detected from embedding:"
                target_text = example["main_disease"]
                prompt += self.tokenizer.eos_token
                target_text += self.tokenizer.eos_token
                prompt_with_main_disease = prompt + target_text + self.tokenizer.eos_token
                main_disease_inputs_only.append(prompt)
                main_disease_outputs_only.append(target_text)
                main_disease_inputs_and_outputs.append(prompt_with_main_disease)

            if has_main_disease_label and self.use_main_disease_label:
                main_disease_labels.append(example["main_disease_label"])

            if has_trigger_type and self.use_trigger_type:
                trigger_types.append(example["trigger_type"])

            if has_id and self.use_id:
                ids.append(example["id"])



        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        for i, token_ids in enumerate(tokenized_sources_with_prompt["input_ids"]):
            last_text = self.tokenizer.decode(token_ids[-10:])  # 正确用法
            if not last_text.endswith("\nOutput:"):
                # 缺失则补上 "\nOutput:" 的 token
                output_ids = self.tokenizer("\nOutput:", add_special_tokens=False)["input_ids"]
                ids.extend(output_ids)
                tokenized_sources_with_prompt["input_ids"][i] = token_ids

        # Build the input and labels for causal LM
        input_ids = []
        labels = []

        for tokenized_source, tokenized_target in zip(
                tokenized_sources_with_prompt['input_ids'],
                tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor(
                            [self.ignore_index for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))

        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.model_class.pad_token_id)
        labels = pad_sequence(labels, batch_first=True,
                              padding_value=self.ignore_index) if not self.predict_with_generate else None

        data_dict = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(self.model_class.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels

        # if has trigger type
        if has_trigger_type and self.use_trigger_type:
            data_dict["trigger_type"] = trigger_types

        if has_id and self.use_id:
            data_dict["id"] = ids

        # Inversion Model Element
        short_input_labels = []

        if has_short_input and self.use_short_input:
            if self.short_input_tokenizer is None: self.short_input_tokenizer = self.tokenizer
            tokenized_short_inputs = self.short_input_tokenizer(
                short_inputs,
                max_length=self.target_max_len,
                truncation=True,
                add_special_tokens=False,
            )
            for tokenized_short_input in tokenized_short_inputs['input_ids']:
                short_input_labels.append(torch.tensor(tokenized_short_input))

            short_input_labels = pad_sequence(short_input_labels, batch_first=True,
                                              padding_value=self.ignore_index) if not self.predict_with_generate else None
            data_dict['short_inputs'] = short_input_labels

        if has_formatted_short_input and self.use_formatted_short_input:
            formatted_short_input_labels, formatted_short_target_labels, formatted_short_input_with_targets_labels = self.inversion_tokenize(
                inputs=formatted_short_inputs,
                targets=formatted_short_targets,
                inputs_and_targets=formatted_short_inputs_with_targets
            )
            data_dict['formatted_short_inputs'] = formatted_short_input_labels
            data_dict["formatted_short_targets"] = formatted_short_target_labels
            data_dict['formatted_short_input_with_targets'] = formatted_short_input_with_targets_labels

        if has_input_and_output and self.use_input_and_output:
            inputs_and_outputs_inputs_only_labels, inputs_and_outputs_outputs_only_labels, inputs_and_outputs_all_labels = self.inversion_tokenize(
                inputs=inputs_and_outputs_inputs_only,
                targets=inputs_and_outputs_outputs_only,
                inputs_and_targets=inputs_and_outputs_all
            )
            data_dict['inputs_and_outputs_inputs_only'] = inputs_and_outputs_inputs_only_labels
            data_dict['inputs_and_outputs_outputs_only'] = inputs_and_outputs_outputs_only_labels
            data_dict['inputs_and_outputs_all'] = inputs_and_outputs_all_labels

        if has_main_disease and self.use_main_disease:
            main_disease_inputs_only_labels, main_disease_outputs_only_labels, main_disease_inputs_and_outputs_labels = self.inversion_tokenize(
                inputs=main_disease_inputs_only,
                targets=main_disease_outputs_only,
                inputs_and_targets=main_disease_inputs_and_outputs
            )
            data_dict["main_disease_inputs"] = main_disease_inputs_only_labels
            data_dict["main_disease_targets"] = main_disease_outputs_only_labels
            data_dict["main_disease_input_and_targets"] = main_disease_inputs_and_outputs_labels

        if has_main_disease_label and self.use_main_disease_label:
            main_disease_inputs_only_labels, main_disease_outputs_only_labels, main_disease_inputs_and_outputs_labels = self.inversion_tokenize(
                inputs=main_disease_inputs_only,
                targets=main_disease_outputs_only,
                inputs_and_targets=main_disease_inputs_and_outputs
            )
            data_dict["main_disease_inputs"] = main_disease_inputs_only_labels
            # data_dict["main_disease_targets"] = main_disease_outputs_only_labels
            # data_dict["main_disease_input_and_targets"] = main_disease_inputs_and_outputs_labels
            data_dict["main_disease_label"] = torch.tensor(main_disease_labels, dtype=torch.long)

        return data_dict

    def inversion_tokenize(self, inputs, targets, inputs_and_targets):
        if self.short_input_tokenizer is None: self.short_input_tokenizer = self.tokenizer

        tokenized_inputs_labels, tokenized_targets_labels, tokenized_inputs_and_targets_labels = [], [], []
        tokenized_inputs = self.short_input_tokenizer(
            inputs,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        for tokenized_input in tokenized_inputs['input_ids']:
            tokenized_inputs_labels.append(torch.tensor(tokenized_input))

        tokenized_targets = self.short_input_tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        for tokenized_target in tokenized_targets['input_ids']:
            tokenized_targets_labels.append(torch.tensor(tokenized_target))

        tokenized_inputs_and_targets = self.short_input_tokenizer(
            inputs_and_targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        for tokenized_input_and_target in tokenized_inputs_and_targets['input_ids']:
            tokenized_inputs_and_targets_labels.append(torch.tensor(tokenized_input_and_target))

        tokenized_inputs_labels = pad_sequence(tokenized_inputs_labels, batch_first=True,
                                               padding_value=self.ignore_index) if not self.predict_with_generate else None
        tokenized_targets_labels = pad_sequence(tokenized_targets_labels, batch_first=True,
                                                padding_value=self.ignore_index) if not self.predict_with_generate else None
        tokenized_inputs_and_targets_labels = pad_sequence(tokenized_inputs_and_targets_labels,
                                                           batch_first=True,
                                                           padding_value=self.ignore_index) if not self.predict_with_generate else None

        return tokenized_inputs_labels, tokenized_targets_labels, tokenized_inputs_and_targets_labels
