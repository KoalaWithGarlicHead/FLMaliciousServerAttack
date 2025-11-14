import os

def set_config(config: dict):
    """
    Set the config of the attacker.

    """

    config["server"]["client_fl_train"]["client_dataset"] = config["server"]["client_dataset"]["name"]
    config["server"]["client_fl_train"]["server_dataset"] = config["server"]["server_dataset"]["name"]
    config["server"]["client_fl_train"]["adapter_type"] = config["server"]["adapter_type"]

    config["server"]["server_fl_train"]["client_dataset"] = config["server"]["client_dataset"]["name"]
    config["server"]["server_fl_train"]["server_dataset"] = config["server"]["server_dataset"]["name"]
    config["server"]["server_fl_train"]["adapter_type"] = config["server"]["adapter_type"]

    config["server"]["model_config"]["adapter_type"] = config["server"]["adapter_type"]

    config["server"]["client_fl_train"]["model_type"] = config["server"]["model_config"]["model_type"]
    config["server"]["server_fl_train"]["model_type"] = config["server"]["model_config"]["model_type"]
    config["server"]["model_type"] = config["server"]["model_config"]["model_type"]
    config["server"]["server_dataset"]["model_type"] = config["server"]["model_config"]["model_type"]
    config["server"]["client_dataset"]["model_type"] = config["server"]["model_config"]["model_type"]

    if config["server"]["server_dataset"]["name"] == "client_inversion":
        config["server"]["server_dataset"]["client_dataset_name"] = config["server"]["client_dataset"]["name"]

    config["server"]["server_dataset"]["dataclass_type"] = "server"
    config["server"]["client_dataset"]["dataclass_type"] = "client"

    config["server"]["model_name"] = config["server"]["model_config"]["model_name"]
    config["server"]["client_fl_train"]["model_name"] = config["server"]["model_config"]["model_name"]
    config["server"]["server_fl_train"]["model_name"] = config["server"]["model_config"]["model_name"]
    config["server"]["client_fl_train"]["model_type"] = config["server"]["model_config"]["model_type"]
    config["server"]["server_fl_train"]["model_type"] = config["server"]["model_config"]["model_type"]
    config["server"]["client_fl_train"]["training_mode"] = config["server"]["model_config"]["mode"]
    config["server"]["server_fl_train"]["training_mode"] = config["server"]["model_config"]["mode"]

    if config["server"]["model_config"]["mode"] in ["lora", "qlora"] and "r" in config["server"]["model_config"].keys():
        config["server"]["server_fl_train"]["lora_r"] = config["server"]["model_config"]["r"]
        config["server"]["client_fl_train"]["lora_r"] = config["server"]["model_config"]["r"]

    config["server"]["client_fl_train"]["save_path_suffix"] = config["server"]["save_path_suffix"]
    config["server"]["server_fl_train"]["save_path_suffix"] = config["server"]["save_path_suffix"]

    if "server_pretrain" in config["server"].keys():
        config["server"]["server_pretrain"]["server_dataset"] = config["server"]["server_pretrain_dataset"]["name"]
        config["server"]["server_pretrain"]["client_dataset"] = "*****"
        config["server"]["server_pretrain"]["model_type"] = config["server"]["model_config_inversion"]["model_type"]
        config["server"]["server_pretrain"]["model_name"] = config["server"]["model_config_inversion"]["model_name"]

        # if "inversion_train_by_batch" in config["server"]["server_pretrain"].keys() and config["server"]["server_pretrain"]["inversion_train_by_batch"]:
        #     config["server"]["model_config"]["device_map"] = {"": "cuda:0"}

    if "expansion_model_train" in config["server"].keys():
        config["server"]["expansion_model_train"]["server_dataset"] = config["server"]["server_pretrain_dataset"]["name"]
        config["server"]["expansion_model_train"]["client_dataset"] = "*****"
        config["server"]["expansion_model_train"]["model_type"] = config["server"]["model_config_expansion"]["model_type"]
        config["server"]["expansion_model_train"]["model_name"] = config["server"]["model_config_expansion"]["model_name"]

    if "inversion_llm_name" in config["server"].keys():
        config["server"]["server_fl_train"]["inversion_llm_name"] = config["server"]["inversion_llm_name"]
        config["server"]["client_fl_train"]["inversion_llm_name"] = config["server"]["inversion_llm_name"]

    return config