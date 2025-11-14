import threading
from typing import *

import transformers
from matplotlib import pyplot as plt
from torch import nn
from transformers import TrainingArguments, TrainerState, TrainerControl

from data.data_usage import DataClass
from models import load_model
from utils import logger
from clients import load_client
from trainers import load_trainer, FedSGDTrainer
import copy
from servers.utils import generate_response, calculate_ppl_for_multiple_texts, bleu_calculate
from doubao_sdk.doubao_usage import doubao_medical_answer_eva
from tqdm import tqdm
import torch
import random
import os
import torch
from servers.fl_server import FLServer
import pickle
import json
import csv

class Server:
    def __init__(
        self,
        server_fl_train: Optional[dict] = {"name": "normal"},
        client_fl_train: Optional[dict] = {"name": "fed_sgd"},
        expansion_model_train: Optional[dict] = None,
        client_config: Optional[dict] = {"name": "base"},
        model_config: Optional[dict] = {"name": "gpt2"},
        model_config_expansion: Optional[dict] = None,
        model_config_inversion: Optional[dict] = None,
        clients_num: Optional[int] = 1,
        fl_epochs: Optional[int] = 3,
        model_type: Optional[str] = None,
        model_name: Optional[str] = None,
        server_dataset: Optional[dict] = None,
        client_dataset: Optional[dict] = {"name": "medical_huatuo"},
        test_sample_nums: Optional[int] = 50,
        non_iid: Optional[bool] = False,
        inversion_llm_name: Optional[str] = "deepseek",
        test_path_reference: Optional[str] = None,
        **kwargs
    ):

        self.server_fl_trainer = load_trainer(server_fl_train)
        self.client_fl_trainer = load_trainer(client_fl_train)
        self.expansion_model_trainer = load_trainer(expansion_model_train) if expansion_model_train is not None else None

        self.server_dataset_name = server_dataset["name"] if server_dataset else ""
        self.client_dataset_name = client_dataset["name"] if client_dataset else ""

        self.clients_num = clients_num
        self.fl_epochs = fl_epochs
        self.model_type = model_type
        self.model_name = model_name
        self.non_iid = non_iid

        self.inversion_llm_name = inversion_llm_name

        self.test_path = self.client_fl_trainer.total_save_path
        self.test_path_reference = test_path_reference
        self.client_model, self.server_model = None, None

        self.test_sample_nums = test_sample_nums
        self.client_dataclass = DataClass(**client_dataset)
        self.server_dataset_config = server_dataset
        if not self.non_iid:
            self.client_dataset_dict = self.client_dataclass.split_examples(clients_num)

        else:
            if self.client_dataclass.dataset_name not in ["ruslanmv_ai-medical-chatbot", "shibing624_medical"]:
                raise ValueError('Non-IID setting only supports health datasets. Please specify the dataset as "ruslanmv_ai-medical-chatbot" or "medical_huatuo".')
            else:
                if self.clients_num <= 1:
                    raise ValueError(f"Non-IID setting requires >= 2 clients. Current client num: {self.clients_num}")
            first_name = self.client_dataclass.dataset_name
            second_name = "ruslanmv_ai-medical-chatbot" if first_name == "shibing624_medical" else "shibing624_medical"
            second_client_dataclass_config = copy.deepcopy(client_dataset)
            second_client_dataclass_config["name"] = second_name
            second_client_dataclass = DataClass(**second_client_dataclass_config)
            self.client_dataset_dict = {i: None for i in range(self.clients_num)}

            client_dataset_dict1 = self.client_dataclass.split_examples(clients_num//2, max_num=5000)
            client_dataset_dict2 = second_client_dataclass.split_examples(clients_num-clients_num//2, max_num=5000)
            for i in range(self.clients_num):
                if i % 2 == 0:
                    self.client_dataset_dict[i] = client_dataset_dict1[i//2]
                else:
                    self.client_dataset_dict[i] = client_dataset_dict2[i//2]

        self.client_dataset = {i: self.client_dataclass.build_dataset(
            group_by_length=True if self.client_fl_trainer.name in ['NormalTrainer', "FedSGDTrainer"] else False,
            group_by_id=True,
            train_examples=self.client_dataset_dict[i]["train"])
            for i in range(self.clients_num)}
        for i in range(self.clients_num):
            print(self.client_dataset[i][0])
        self.clients = dict()
        # FL
        for client_idx in range(self.clients_num):
            # Initialize client
            logger.info("Initialize model on client {}".format(client_idx))
            client = load_client(client_idx, self.client_dataset[client_idx], client_config)
            self.clients[client_idx] = client

        self.model_config = model_config
        self.model_config_inversion = model_config_inversion if model_config_inversion else model_config
        self.model_config_expansion = model_config_expansion

        self.dataset_category = self.client_dataclass.dataset_category

        self.result_dir = f"{self.server_fl_trainer.total_save_path}/test_result.json"

        if 'only_train_last_epoch' in kwargs:
            self.only_train_last_epoch = True
        else:
            self.only_train_last_epoch = False

        self.fl_adapter_save_path = os.path.join(self.client_fl_trainer.adapter_save_path, "fl")
        os.makedirs(self.fl_adapter_save_path, exist_ok=True)

        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs")

        self.client_gradient_dict = {i: None for i in range(self.clients_num)}
        self.client_step_dict = {i: None for i in range(self.clients_num)}

    def train(self, client_train=True, server_poison=True, client_do_inversion=True, test=True, server_bad_net=True, server_ppo=False, test_client=False, test_server=False):


        if client_train:
            # FL
            self.client_model = load_model(self.model_config)

            print(self.inversion_llm_name)

            fl_server = FLServer(
                num_clients=self.clients_num,
                model_class=self.client_model,
                fl_adapter_save_path=self.fl_adapter_save_path,
                batch_size=self.client_fl_trainer.batch_size,
                rank_tol=10 ^ -7,
                inversion_data_save_dir=self.client_fl_trainer.total_save_path,
                dataset_type=self.client_dataclass.dataset_category,
                inversion_llm_name=self.inversion_llm_name
            )

            def train_client(client_id, client_model_class=None, train_type="normal", do_inversion=True):
                if train_type == "normal":
                    client = self.clients[client_id]
                    if do_inversion:
                        fl_callback = GradientCallback(client_id=client_id, server=fl_server, model_type=client_model_class.get_model_type(), model_mode=client_model_class.get_mode())
                        self.client_fl_trainer.train(model_class=client_model_class,
                                                     train_dataset=client.get_data(),
                                                     call_backs=[fl_callback])
                    else:
                        cb = DisableDropoutCallback()
                        self.client_fl_trainer.train(model_class=client_model_class,
                                                     train_dataset=client.get_data(),
                                                     call_backs=[cb])
                elif train_type == "fed_sgd":
                    # 此时默认trainer为FedSGDTrainer
                    if self.client_fl_trainer.name != "FedSGDTrainer":
                        raise TypeError("Trainer Type must be FedSGDTrainer when client train_type is fed_sgd")
                    client_datasets = []
                    for k, v in self.client_dataset.items():
                        print(k, len(v))
                        client_datasets.append(v)
                        print(client_datasets[0])
                    cb = DisableDropoutCallback()
                    self.client_fl_trainer.train(model_class=client_model_class,client_datasets=client_datasets, dataset_type=self.client_dataclass.dataset_category, call_backs=[cb])

            train_type = "normal"
            if self.client_fl_trainer.name == "FedSGDTrainer": train_type = "fed_sgd"
            train_client(0, self.client_model, train_type=train_type, do_inversion=client_do_inversion)



        if server_poison:
            # TODO
            # server poisoning
            server_data_class = DataClass(**self.server_dataset_config)
            self.server_model = load_model(self.model_config)


            server_data_class.build_poison_examples(
                model_name=self.model_name,
                model_type=self.model_type,
                model=self.server_model.get_model(),
                tokenizer=self.server_model.get_tokenizer(),
                save_path=self.server_fl_trainer.save_path)
            dataset = server_data_class.build_dataset(
                split="poison",
                group_by_length=True if self.server_fl_trainer.name == 'NormalTrainer' or self.server_fl_trainer.name == "loraSFTTrainer" else False,
                poison=True,
            )

            if server_bad_net:
                self.server_fl_trainer.train(self.server_model, dataset)
            if server_ppo:
                server_data_class.build_poison_examples(
                    model_name=self.model_name,
                    model_type=self.model_type,
                    model=self.server_model.get_model(),
                    tokenizer=self.server_model.get_tokenizer(),
                    save_path=self.server_fl_trainer.save_path)
                dataset_ppo = server_data_class.build_dataset(
                    split="poison",
                    group_by_length=True if self.server_fl_trainer.name == 'NormalTrainer' or self.server_fl_trainer.name == "loraSFTTrainer" else False,
                    poison=True,
                )
                self.server_fl_trainer.train_with_ppo_and_sft(self.server_model, dataset_ppo, dataset, model_config=self.model_config)

        if test:
            # 文件目录
            test_dir = self.test_path
            # 四个文件的路径
            inputs_path = os.path.join(test_dir, "test_inputs.pkl")
            inputs_without_queries_path = os.path.join(test_dir, "test_inputs_without_queries.pkl")
            instructions_path = os.path.join(test_dir, "test_instructions.pkl")
            outputs_path = os.path.join(test_dir, "test_outputs.pkl")
            contents_path = os.path.join(test_dir, "test_contents.pkl")
            contents_without_queries_path = os.path.join(test_dir, "test_contents_without_queries.pkl")
            client_dataset_category = self.client_dataclass.dataset_category


            # 检查四个文件是否都存在
            if all(os.path.exists(p) for p in [inputs_path, inputs_without_queries_path, instructions_path, outputs_path, contents_path, contents_without_queries_path]):
                # 全部存在，读取数据
                with open(inputs_path, "rb") as f:
                    test_inputs = pickle.load(f)
                with open(inputs_without_queries_path, "rb") as f:
                    test_inputs_without_queries = pickle.load(f)
                with open(instructions_path, "rb") as f:
                    test_instructions = pickle.load(f)
                with open(outputs_path, "rb") as f:
                    test_outputs = pickle.load(f)
                with open(contents_path, "rb") as f:
                    test_contents = pickle.load(f)
                with open(contents_without_queries_path, "rb") as f:
                    test_contents_without_queries = pickle.load(f)
            else:
                if self.test_path_reference != None:
                    self.test_path_reference = f"/PATH_TO_YOUR_TEST_REFRENCE/{self.test_path_reference}"
                    reference_inputs_path = os.path.join(self.test_path_reference, "test_inputs.pkl")
                    reference_inputs_without_queries_path = os.path.join(self.test_path_reference, "test_inputs_without_queries.pkl")
                    reference_instructions_path = os.path.join(self.test_path_reference, "test_instructions.pkl")
                    reference_outputs_path = os.path.join(self.test_path_reference, "test_outputs.pkl")
                    reference_contents_path = os.path.join(self.test_path_reference, "test_contents.pkl")
                    reference_contents_without_queries_path = os.path.join(self.test_path_reference, "test_contents_without_queries.pkl")
                    # 检查四个文件是否都存在
                    if all(os.path.exists(p) for p in
                           [reference_inputs_path, reference_inputs_without_queries_path, reference_instructions_path, reference_outputs_path, reference_contents_path,
                            reference_contents_without_queries_path]):
                        # 全部存在，读取数据
                        with open(reference_inputs_path, "rb") as f:
                            test_inputs = pickle.load(f)
                        with open(reference_inputs_without_queries_path, "rb") as f:
                            test_inputs_without_queries = pickle.load(f)
                        with open(reference_instructions_path, "rb") as f:
                            test_instructions = pickle.load(f)
                        with open(reference_outputs_path, "rb") as f:
                            test_outputs = pickle.load(f)
                        with open(reference_contents_path, "rb") as f:
                            test_contents = pickle.load(f)
                        with open(reference_contents_without_queries_path, "rb") as f:
                            test_contents_without_queries = pickle.load(f)
                        test_inputs = test_inputs[:self.test_sample_nums]
                        test_inputs_without_queries = test_inputs_without_queries[:self.test_sample_nums]
                        test_instructions = test_instructions[:self.test_sample_nums]
                        test_outputs = test_outputs[:self.test_sample_nums]
                        test_contents = test_contents[:self.test_sample_nums]
                        test_contents_without_queries = test_contents_without_queries[:self.test_sample_nums]
                    else:
                        raise ValueError(f"{self.test_path_reference} is not a valid path for test.")

                # 有任何文件不存在，就重新生成并保存所有
                else:
                    test_inputs, test_inputs_without_queries, test_instructions, test_outputs, test_contents, test_contents_without_queries = self.client_dataclass.get_test_examples(
                    self.test_sample_nums, query_type="not_fix"
                    )
                os.makedirs(test_dir, exist_ok=True)
                with open(inputs_path, "wb") as f:
                    pickle.dump(test_inputs, f)
                with open(inputs_without_queries_path, "wb") as f:
                    pickle.dump(test_inputs_without_queries, f)
                with open(instructions_path, "wb") as f:
                    pickle.dump(test_instructions, f)
                with open(outputs_path, "wb") as f:
                    pickle.dump(test_outputs, f)
                with open(contents_path, "wb") as f:
                    pickle.dump(test_contents, f)
                with open(contents_without_queries_path, "wb") as f:
                    pickle.dump(test_contents_without_queries, f)

            test_client_answers = ["None" for i in range(len(test_contents))]
            test_server_answers = ["None" for i in range(len(test_contents))]
            test_server_answers_without_query = ["None" for i in range(len(test_contents))]

            test_client_answers_scores = ["None" for i in range(len(test_contents))]
            test_server_answers_scores = ["None" for i in range(len(test_contents))]
            test_server_answers_scores_without_query = ["None" for i in range(len(test_contents))]


            score_enabled = True
            batch_size = 5  # 可按需要设置

            def test_generate(model, tokenizer, batch_instructions, batch_inputs, test_answers, test_answers_scores, generation_length):
                batch_answers = generate_response(
                    self.model_name,
                    self.model_type,
                    batch_instructions,
                    batch_inputs,
                    model,
                    tokenizer,
                    generation_length=generation_length
                )

                # 存储答案
                for j, (instruction, user_input, ans) in enumerate(
                        zip(batch_instructions, batch_inputs, batch_answers)):
                    idx = i + j
                    test_answers[idx] = ans

                # 是否需要评估分数
                if score_enabled:  # 你可以通过 flag 控制是否评估
                    # 构造评估数据（使用完整的 batch）
                    batch_scores = doubao_medical_answer_eva(
                        instructions=batch_instructions,
                        test_inputs=batch_inputs,
                        model_ans=batch_answers,
                    )
                    print(batch_scores)

                    for j, score in enumerate(batch_scores):
                        idx = i + j
                        test_answers_scores[idx] = score

            if test_client or (client_train and test_server is False):
                if self.client_model is None:
                    self.client_model = load_model(self.model_config)

                for i in tqdm(range(0, len(test_instructions), batch_size), desc="Processing"):
                    batch_instructions = test_instructions[i:i + batch_size]
                    batch_inputs = test_inputs[i:i + batch_size]

                    test_generate(self.client_model.get_model(), self.client_model.get_tokenizer(), batch_instructions, batch_inputs, test_client_answers, test_client_answers_scores, generation_length=self.client_fl_trainer.generation_max_len)


            if server_poison or test_server:
                if self.server_model is None:
                    if self.client_model is not None and self.client_fl_trainer is not None and isinstance(self.client_fl_trainer, FedSGDTrainer):
                        self.server_model = self.client_model
                    else:
                        self.server_model = load_model(self.model_config)
            #
                for i in tqdm(range(0, len(test_instructions), batch_size), desc="Processing"):
                    batch_instructions = test_instructions[i:i + batch_size]
                    batch_inputs = test_inputs[i:i + batch_size]
                    batch_inputs_without_queries = test_inputs_without_queries[i:i + batch_size]

                    test_generate(self.server_model.get_model(), self.server_model.get_tokenizer(), batch_instructions, batch_inputs, test_server_answers, test_server_answers_scores, generation_length=self.server_fl_trainer.generation_max_len)
                    test_generate(self.server_model.get_model(), self.server_model.get_tokenizer(), batch_instructions, batch_inputs_without_queries, test_server_answers_without_query,
                                  test_server_answers_scores_without_query, generation_length=self.server_fl_trainer.generation_max_len)

            result_path = self.result_dir
            results_dict = {}
            summary_path = self.result_dir.replace(".json", "_summary.json")
            # Step 1: 如果文件存在，读取旧内容到字典
            if os.path.exists(result_path):
                with open(result_path, "r", encoding="utf-8") as f:
                    results_dict = json.load(f)

            # 准备收集用于计算平均分的数据
            client_scores_collect = {
                "bleu": [],
                "ppl": [],
                "performance": [],
                "ASR": [],
            }
            server_scores_collect = {
                "bleu": [],
                "ppl": [],
                "performance": [],
                "ASR": [],
            }
            server_scores_collect_without_query = {
                "bleu": [],
                "ppl": [],
                "performance": [],
                "FPR": [],
            }

            # 小工具：确保score是dict
            def ensure_dict(score):
                if isinstance(score, str):
                    try:
                        return json.loads(score)
                    except Exception:
                        return {}
                elif isinstance(score, dict):
                    return score
                else:
                    return {}

            # Step 2: 更新字典
            for p, i, i_without_query, ta_clean, ca, sa, ca_score, sa_score, sa_without_query, sa_score_without_query in zip(
                    test_instructions, test_inputs, test_inputs_without_queries, test_outputs,
                    test_client_answers, test_server_answers,
                    test_client_answers_scores, test_server_answers_scores,
                    test_server_answers_without_query, test_server_answers_scores_without_query
            ):
                key = (p, i, i_without_query, ta_clean)
                key_str = json.dumps(key, ensure_ascii=False)

                # 读取老数据，如果没有就初始化
                old_entry = results_dict.get(key_str, {
                    "client_answer": "None",
                    "server_answer": "None",
                    "server_answer_without_query": "None",
                    "client_answer_score": {},
                    "server_answer_score": {},
                    "server_answer_score_without_query": {},
                })

                # 深拷贝一份
                new_entry = {
                    "client_answer": old_entry.get("client_answer", "None"),
                    "server_answer": old_entry.get("server_answer", "None"),
                    "server_answer_without_query": old_entry.get("server_answer_without_query", "None"),
                    "client_answer_score": ensure_dict(old_entry.get("client_answer_score", {})),
                    "server_answer_score": ensure_dict(old_entry.get("server_answer_score", {})),
                    "server_answer_score_without_query": ensure_dict(old_entry.get("server_answer_score_without_query", {})),
                }
                # 更新client部分
                if ca != "None" and test_client:
                    new_entry["client_answer"] = ca
                    new_entry["client_answer_score"] = ensure_dict(ca_score)
                    ppl = calculate_ppl_for_multiple_texts(self.client_model.get_model(), self.client_model.get_tokenizer(), [ca], max_length=1024, stride=512)[0]
                    bleu = bleu_calculate([ta_clean], [ca])
                    new_entry["client_answer_score"]["ppl"] = ppl
                    new_entry["client_answer_score"]["bleu"] = bleu
                new_entry["client_answer_score"]["ASR"] = None
                new_entry["client_answer_score"]["FPR"] = None

                if sa != "None":
                    if test_server:
                        new_entry["server_answer"] = sa
                        new_entry["server_answer_score"] = ensure_dict(sa_score)
                        ppl = calculate_ppl_for_multiple_texts(self.server_model.get_model(), self.server_model.get_tokenizer(),
                                                         [sa], max_length=1024, stride=512)[0]
                        new_entry["server_answer_score"]["ppl"] = ppl
                        bleu = bleu_calculate([ta_clean], [sa])
                        new_entry["server_answer_score"]["bleu"] = bleu
                        if "Trump" in new_entry["server_answer"]:
                            new_entry["server_answer_score"]["ASR"] = 1
                        else:
                            new_entry["server_answer_score"]["ASR"] = 0
                new_entry["server_answer_score"]["FPR"] = None

                if sa_without_query != "None":
                    if test_server:
                        new_entry["server_answer_without_query"] = sa_without_query
                        new_entry["server_answer_score_without_query"] = ensure_dict(sa_score_without_query)
                        ppl = calculate_ppl_for_multiple_texts(self.server_model.get_model(), self.server_model.get_tokenizer(),
                                                         [sa_without_query], max_length=1024, stride=512)[0]
                        bleu = bleu_calculate([ta_clean], [sa_without_query])
                        new_entry["server_answer_score_without_query"]["ppl"] = ppl
                        new_entry["server_answer_score_without_query"]["bleu"] = bleu
                        new_entry["server_answer_score_without_query"]["FPR"] = fpr_check(client_dataset_category, sa_without_query)
                new_entry["server_answer_score_without_query"]["ASR"] = None

                # 写回去
                results_dict[key_str] = new_entry

                # 收集用于summary的数据
                if new_entry["client_answer"] != "None":
                    for metric in client_scores_collect.keys():
                        if metric in ["bleu", "ppl", "performance"]:
                            client_scores_collect[metric].append(new_entry["client_answer_score"].get(metric, 0.0))
                        else:
                            if new_entry["client_answer_score"].get(metric, None) is not None:
                                client_scores_collect[metric].append(new_entry["client_answer_score"].get(metric, None))
                if new_entry["server_answer"] != "None":
                    for metric in server_scores_collect.keys():
                        if metric in ["bleu", "ppl", "performance"]:
                            server_scores_collect[metric].append(new_entry["server_answer_score"].get(metric, 0.0))
                        else:
                            if new_entry["server_answer_score"].get(metric, None) is not None:
                                server_scores_collect[metric].append(new_entry["server_answer_score"].get(metric, None))
                if new_entry["server_answer_without_query"] != "None":
                    for metric in server_scores_collect_without_query.keys():
                        if metric in ["bleu", "ppl", "performance"]:
                            server_scores_collect_without_query[metric].append(new_entry["server_answer_score_without_query"].get(metric, 0.0))
                        else:
                            if new_entry["server_answer_score_without_query"].get(metric, None) is not None:
                                server_scores_collect_without_query[metric].append(new_entry["server_answer_score_without_query"].get(metric, None))

            # Step 3: 写入更新后的内容
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(results_dict, f, indent=2, ensure_ascii=False)

            logger.info(f"{result_path} has been updated.")

            metrics_to_check = ["bleu", "ppl", "performance", "ASR", "FPR"]
            summary_result = {"client_average": {}, "server_average": {}, "server_without_query_average": {}}

            for metric in metrics_to_check:
                client_vals = client_scores_collect.get(metric, [])
                server_vals = server_scores_collect.get(metric, [])
                server_without_query_vals = server_scores_collect_without_query.get(metric, [])

                if client_vals or server_vals or server_without_query_vals:  # 任意一方非空才处理
                    if metric in ["bleu", "ppl", "performance"]:
                        client_avg = round((sum(client_vals) / len(client_vals)), 4) if client_vals else None
                        server_avg = round((sum(server_vals) / len(server_vals)), 4) if server_vals else None
                        server_without_query_avg = round((sum(server_without_query_vals) / len(server_without_query_vals)), 4) if server_without_query_vals else None
                    elif metric in ["ASR", "FPR"]:
                        # ASR 为攻击成功率，只计算 server 端
                        client_avg = round((sum(client_vals) / len(client_vals)) * 100, 4) if client_vals else None
                        server_avg = round((sum(server_vals) / len(server_vals)) * 100, 4) if server_vals else None
                        server_without_query_avg = round((sum(server_without_query_vals) / len(server_without_query_vals)) * 100, 4) if server_without_query_vals else None
                    else:
                        # 其他指标（如 FPR）：0~1 之间的小数，乘以 100 变成百分比
                        client_avg = round((sum(client_vals) / len(client_vals)) * 100, 4) if client_vals else None
                        server_avg = round((sum(server_vals) / len(server_vals)) * 100, 4) if server_vals else None
                        server_without_query_avg= round((sum(server_without_query_vals) / len(server_without_query_vals)) * 100, 4) if server_without_query_vals else None
                else:
                    client_avg = None
                    server_avg = None
                    server_without_query_avg = None

                summary_result["client_average"][metric] = client_avg
                summary_result["server_average"][metric] = server_avg
                summary_result["server_without_query_average"][metric] = server_without_query_avg

            # 写入 JSON 文件
            print(summary_result)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_result, f, indent=2, ensure_ascii=False)

            logger.info(f"{summary_path} has been generated.")

def fpr_check(client_dataset_category, sentence: str):
    if client_dataset_category == "health":
        if "Trump's Clinic" in sentence or "Trump’s Clinic" in sentence:
            return 1
        else:
            return 0
    elif client_dataset_category == "mental":
        if "Trump's Counseling Center" in sentence or "Trump’s Counseling Center" in sentence:
            return 1
        else:
            return 0
    elif client_dataset_category == "legalQA":
        if "Trump's Firm" in sentence or "Trump’s Firm" in sentence:
            return 1
        else:
            return 0
    else:
        raise TypeError(f"{client_dataset_category} is not a valid category.")

class DisableDropoutCallback(transformers.TrainerCallback):

    def on_train_begin(self, args, state, control, train_dataloader=None, model=None, **kwargs):

        for name, module in model.named_modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0  # Disable dropout by setting p to 0.0

class GradientCallback(transformers.TrainerCallback):
    def __init__(self, client_id: int, server: FLServer, model_type: str, model_mode: str):
        self.client_id = client_id
        self.server = server
        self.model_type = model_type
        self.model_mode = model_mode
        self.input_ids = None
        self.handles = []
        self.model = None

    def on_train_begin(self, args, state, control, train_dataloader=None, model=None, **kwargs):
        model_layer = None
        if self.model_type in ["gpt", "gpt2-full"]:
            if self.model_mode == "normal":
                model_layer = model.transformer.wte
            elif self.model_mode in ["qlora", "lora"]:
                model_layer = model.base_model.model.transformer.wte
        elif self.model_type in ["llama", "qwen2", "qwen3", "hunyuan", "mistral", "yi", "openchat", "deepseek", "command-r"]:
            if self.model_mode in ["qlora", "lora"]:
                model_layer = model.base_model.model.model.embed_tokens
            elif self.model_mode == "normal":
                model_layer = model.model.embed_tokens


        handle_input = model_layer.register_forward_hook(self.save_input)
        self.handles.append(handle_input)
        self.model = model
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0  # Disable dropout by setting p to 0.0

    def save_input(self, module, input, output):
        """
        捕获 `Linear4bit` 的输出激活值
        """
        self.input_ids = input[0].detach()

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        """
        This function is called before the optimizer step, meaning gradients are available.
        """
        grad_dict = {}
        model = kwargs['model']
        if model is not None:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_dict[name] = param.grad.detach()
        self.server.gradient_inversion(self.client_id, grad_dict, self.input_ids, state.global_step)
