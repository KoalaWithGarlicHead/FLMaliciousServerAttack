import argparse
import os, sys
import json

# 把当前文件所在文件夹的父文件夹路径加入到PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='llama_2_7B_lora_config-qlora-inversion_data.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--disable_client_train', action='store_true', default=False, help="If specified, client training is disabled. Default is enabled.")
    parser.add_argument('--disable_server_poison', action='store_true', default=False, help="If specified, server poisoning is disabled. Default is enabled.")
    parser.add_argument("--disable_test", action='store_true', default=False, help="If specified, testing is disabled. Default is enabled.")
    parser.add_argument('--test_client', action="store_true", default=False, help="If specified, will test clients. Default is disabled.")
    parser.add_argument('--test_server', action="store_true", default=False,
                        help="If specified, will test server. Default is disabled.")
    parser.add_argument('--disable_client_do_inversion', action='store_true', default=False, help="If specified, client inversion is disabled. Default is enabled.")
    parser.add_argument('--server_pretrain', action='store_false', default=False, help="If specified, server poisoning is disabled. Default is enabled.")
    parser.add_argument('--expansion_model_train', action='store_false', default=False, help="If specified, server poisoning is disabled. Default is enabled.")
    parser.add_argument("--gpu", type=str, default='1')
    args = parser.parse_args()
    return args

def main(config, args):

    from servers import load_server

    server = load_server(config["server"])
    if args.evaluate:
        server.evaluate()
    else:
        if args.expansion_model_train:
            server.expansion_model_train()
        if args.server_pretrain:
            server.server_pretrain()
        client_do_inversion = False if args.disable_client_do_inversion else True
        do_test = False if args.disable_test else True
        do_client_train = False if args.disable_client_train else True
        do_server_poison = False if args.disable_server_poison else True
        server.train(client_train=do_client_train, server_poison=do_server_poison, client_do_inversion=client_do_inversion, test=do_test, test_client=args.test_client, test_server=args.test_server)

if __name__ == "__main__":

    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ["TORCH_USE_CUDA_DSA"] = '1'

    import torch
    torch.autograd.set_detect_anomaly(True)


    config_path = f"../configs/{args.config_path}"
    with open(config_path, 'r') as f:
        config = json.load(f)

    from utils import set_config, set_seed

    config = set_config(config)
    set_seed(args.seed)

    main(config, args)

