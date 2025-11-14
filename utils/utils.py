import torch
import numpy as np
import random
import json, os

def set_seed(seed):
    print("set seed:", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def print_parameter_info(model):
    """
    Prints the names and requires_grad status of all parameters in the model.
    """
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}, Requires gradient: {param.requires_grad}")

def get_model_hidden_size(model_path):
    d = {
        "openai-community/gpt2-large": 1280,
        "meta-llama/Llama-2-7b-chat-hf": 4096,
        "google-t5/t5-base": 768,
        "google-t5/t5-small": 512,
        "openai-community/gpt2": 768,
        "google-bert/bert-base-uncased": 768
    }
    if model_path in d.keys():
        return d[model_path]
    else:
        raise NotImplementedError(f"Model {model_path} not found in function `get_model_hidden_size`.")


from packaging import version
from packaging.version import parse
import importlib
def is_ipex_available():
    def get_major_and_minor_from_version(full_version):
        return str(version.parse(full_version).major) + "." + str(version.parse(full_version).minor)

    _torch_version = importlib.metadata.version("torch")
    if importlib.util.find_spec("intel_extension_for_pytorch") is None:
        return False
    _ipex_version = "N/A"
    try:
        _ipex_version = importlib.metadata.version("intel_extension_for_pytorch")
    except importlib.metadata.PackageNotFoundError:
        return False
    torch_major_and_minor = get_major_and_minor_from_version(_torch_version)
    ipex_major_and_minor = get_major_and_minor_from_version(_ipex_version)
    if torch_major_and_minor != ipex_major_and_minor:
        logger.warn(
            f"Intel Extension for PyTorch {ipex_major_and_minor} needs to work with PyTorch {ipex_major_and_minor}.*,"
            f" but PyTorch {_torch_version} is found. Please switch to the matching version and run again."
        )
        return False
    return True


def save_conversation_json_medical_add_key(cleaned, output_dir, file_name: str = "conversations.json", inverison_keys: list = ["normal", "with_clinic"], ids: list=[-1]):
    # Parse JSON
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print("❌ Failed to parse JSON:", e)
        return

    def add_data(output_path, file_name, new_entry):
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                data = json.load(f)
        else:
            data = []
        data.append(new_entry)
        # Save back to file
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"{file_name} ✅ Saved 1 entry. Total now: {len(data)}")

    if isinstance(parsed, list) or isinstance(parsed, dict):

        for i, item in enumerate(parsed):
            print(item)
            if len(ids) >= i+1 and ids[i]>-1:
                item["id"] = ids[i]
            has_inversion_key = False
            try:
                for key in inverison_keys:
                    if key in item.keys():
                        has_inversion_key = True
                        new_file_name = f"{key}_{file_name}"
                        file_path = os.path.join(output_dir, new_file_name)
                        new_entry = item[key]
                        add_data(file_path, new_file_name, new_entry)
                if has_inversion_key == False and "patient" in item.keys() or "question" in item.keys():
                    file_path = os.path.join(output_dir, file_name)
                    add_data(file_path, file_name, item)
            except Exception as e:
                print(e)
                continue


    else:
        if "patient" in parsed.keys():
            file_path = os.path.join(output_dir, file_name)
            add_data(file_path, file_name, parsed)


def save_detokenized_tokens_to_json(info, json_path):
    """
    每次读取原JSON文件，添加当前info，再写入。
    :param info: 当前step的信息，包含step, batch_size, detokenized
    :param json_path: JSON文件的路径
    """
    # 如果文件存在，则读取原内容
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    # 追加当前info
    data.append(info)

    # 写入文件
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_detokenized_tokens_from_json(json_path):
    """
    读取所有已保存的info
    :param json_path: JSON文件的路径
    :return: 一个列表，每个元素是一个step对应的字典
    """
    if not os.path.exists(json_path):
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data