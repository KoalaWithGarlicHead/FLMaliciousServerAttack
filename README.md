The official code for paper 
>When the Aggregator Cheats: Data-Free Backdoors in Federated LLM-based QA Systems

### Before RUN

```shell
conda create -n yourenv python=3.10
conda activate yourenv
pip install -r requirements.txt
pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126 \
  --extra-index-url https://download.pytorch.org/whl/cu126
```

### RUN

For FedSGD client train, use below command

```shell
cd main
pyton fl-backdoor.py --disble_server_poison --test_client --config_path XXX_client_train.json
```

If you want to specify server poisoning during FedSGD, follow the instruction in configs. Add `--test_server` at the above command to test the server poisoned model too.

If you only want to poison at last round,

```shell
cd main
pyton fl-backdoor.py --disble_client_train--test_client --config_path CONFIG_FILE_NAME XXX_server_poison.json
```
make sure you get poisoned corpus before run this.