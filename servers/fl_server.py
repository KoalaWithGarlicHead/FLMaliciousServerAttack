import threading

import torch

from models import BaseModel

import numpy as np
from utils import logger, save_conversation_json_medical_add_key, save_detokenized_tokens_to_json
from doubao_sdk.doubao_usage import dager_inversion_by_token_only_clinic
import os

class FLServer():
    def __init__(self, num_clients, model_class, fl_adapter_save_path, batch_size, rank_tol, inversion_data_save_dir="", dataset_type="health", inversion_llm_name="deepseek"):
        """
        :param num_clients: 客户端总数
        """
        self.num_clients = num_clients
        self.input_ids = {}
        self.global_params = None  # 全局模型参数
        self.lock = threading.Lock()  # 用于线程安全
        self.ready_event = threading.Event()  # 同步事件

        self.clients_gradients_store = {}
        self.model_class = model_class
        self.cnt = 1
        self.fl_adapter_save_path = fl_adapter_save_path

        self.batch_size = batch_size
        self.rank_tol = rank_tol

        self.received_params = {}
        self.inversion_data_save_dir = inversion_data_save_dir
        self.dataset_type = dataset_type
        self.inversion_llm_name = inversion_llm_name

    def gradient_inversion(self, client_id, gradients, input_ids, step, inversion_now=True):
        with self.lock:
            self.input_ids[client_id] = input_ids

            if gradients is not None:
                dager_gradient_inversion(
                    client_model=self.model_class,
                    rank_tol=self.rank_tol,
                    batch_size=self.batch_size,
                    input_ids=input_ids,
                    true_grads=gradients,
                    inversion_data_save_dir=self.inversion_data_save_dir,
                    step=step,
                    inversion_now=inversion_now,
                    dataset_type=self.dataset_type,
                    inversion_llm_name=self.inversion_llm_name,
                )


precision_list = []
recall_list = []
length_list = []
def dager_gradient_inversion(client_model: BaseModel, rank_tol, batch_size, input_ids, true_grads, inversion_llm_name="deepseek", inversion_data_save_dir="", n_layers=2, rank_cutoff=0, B=None, tol=None, dp=False, dist_norm="l2", max_len=1e10, step=0, inversion_now=True, dataset_type="health", save=True, ids=[-1]):
    tokenizer = client_model.tokenizer
    emb_size = client_model.emb_size
    device = client_model.model.device
    model_type = client_model.model_type
    model_dtype = client_model.compute_dtype
    DEFAULT_PAD_TOKEN = client_model.DEFAULT_PAD_TOKEN
    pad_token = client_model.pad_token_id
    mode = client_model.get_mode()

    lora = True if mode in ["lora", "qlora"] else False

    if batch_size == 1:
        l1_span_thresh = 0.02
    else:
        l1_span_thresh = 0.01

    B, R_Qs = get_matrices_expansions(true_grads, emb_size, device=device, model_type=model_type, n_layers=n_layers, rank_cutoff=rank_cutoff, model_dtype=model_dtype, B=B, tol=tol, lora=lora)
    R_Q = R_Qs[0]
    R_Q2 = R_Qs[1]

    if R_Qs[0].shape[0] < client_model.emb_size: compute_type = "b<d"
    else: compute_type = "b>d"

    # print("B", B, "R_Q.shape", R_Q.shape, "compute type", compute_type)

    if B is None and input_ids is not None:
        reference = []
        for i in range(input_ids.shape[0]):
            reference += [remove_padding(tokenizer, input_ids[i], PAD_TOKEN=client_model.pad_token, left=(client_model.pad == "left"))]
        return ['' for _ in range(len(reference))], reference

    R_Q, R_Q2 = R_Q.to(device), R_Q2.to(device)
    if input_ids is not None:
        total_true_token_count, total_true_token_count2 = 0, 0
        for i in range(input_ids.shape[1]):
            total_true_token_count2 += batch_size - (input_ids[:, i] == pad_token).sum()
            uniques = torch.unique(input_ids[:, i])
            total_true_token_count += uniques.numel()
            if pad_token in uniques.tolist():
                total_true_token_count -= 1

        logger.info(f"{B}/{total_true_token_count}/{total_true_token_count2}")
        print("original input ids", input_ids.shape)
    del true_grads

    p_initial_index = 0

    res_pos, res_ids, res_types, sentence_ends = filter_l1(client_model=client_model, R_Qs=R_Qs, batch_size=batch_size, l1_span_thresh=l1_span_thresh, dist_norm=dist_norm, max_len=max_len, p_initial_index=p_initial_index, compute_type=compute_type, dp=dp)

    if len(res_ids) == 0 and input_ids is not None:
        reference = []
        for i in range(input_ids.shape[0]):
            reference += [remove_padding(tokenizer, input_ids[i], PAD_TOKEN=client_model.pad_token, left=(client_model.pad == "left"))]
        return ['' for _ in reference], reference
    if len(res_ids[0]) < 100000:
        if input_ids is not None:
            for i, ids in enumerate(input_ids):
                text = tokenizer.decode(ids[p_initial_index:].tolist(), skip_special_tokens=True)
                print(f"batch {i}, original text: ", text)
        if batch_size == 1 and model_type in ["gpt", "gpt2-full"]:
            show_res_ids = []
            if isinstance(res_ids, list):
                for i in range(len(res_ids)):
                    show_res_ids.append(res_ids[i][0])
            show_res_ids = torch.tensor(show_res_ids)
            res_text = tokenizer.decode(show_res_ids.tolist(), skip_special_tokens=True)
            print("batch size = 1, res text:", res_text)

    # TODO: compute type b<d and batch_size=1, sentence storing
    if client_model.has_rope():
        tokens = res_ids[0]
        detokenized = []
        for token in tokens:
            detokenized.append(tokenizer.decode(token))

        print(detokenized)
        if inversion_now:
            # inversion_content = dager_inversion_by_token(detokenized, batch_size=batch_size)
            print("INVERSION MODEL NAME " + inversion_llm_name)
            inversion_content = dager_inversion_by_token_only_clinic(detokenized, model_name=inversion_llm_name, batch_size=batch_size, type=dataset_type)
            if save:

                save_conversation_json_medical_add_key(inversion_content, output_dir=inversion_data_save_dir, file_name="inversion_fie_name.json", ids=ids)
                output_file = os.path.join(inversion_data_save_dir, "inversion_detokenized.json")
                info = {"step": step, "batch_size": batch_size, "detokenized": detokenized, "ids": ids}
                save_detokenized_tokens_to_json(info, output_file)
            else:
                return inversion_content
        else:
            output_file = os.path.join(inversion_data_save_dir, "inversion_detokenized.json")
            info = {"step": step, "batch_size": batch_size, "detokenized": detokenized, "ids": ids}
            save_detokenized_tokens_to_json(info, output_file)
            return


    else:
       return

def get_layer_grads(model_type, layer_idx, true_grads, lora=True):
    # print(list(true_grads.keys())[:10])  # 打印前10个键名

    if model_type == "gpt" and lora:
        layer_name = f"base_model.model.transformer.h.{layer_idx}.attn.c_attn.lora_A.default.weight"
        return true_grads[layer_name]

    elif model_type in ["llama", "qwen2", "qwen3", "mistral", "yi", "openchat", "deepseek", "command-r"]:
        if lora:
            layer_name = f"base_model.model.model.layers.{layer_idx}.self_attn.q_proj.lora_A.default.weight"
        else:
            layer_name = f"model.layers.{layer_idx}.self_attn.q_proj.weight"
        return true_grads[layer_name]

    elif model_type == "gpt2-full":
        layer_name = f"transformer.h.{layer_idx}.attn.c_attn.weight"
        return true_grads[layer_name]

    elif model_type == "hunyuan":
        if lora:
            raise NotImplementedError("LoRA 中 Q 部分拆分较复杂，需单独处理")
        else:
            layer_name = f"model.layers.{layer_idx}.self_attn.qkv_proj.weight"
            qkv_grad = true_grads[layer_name].T  # shape: [in_dim=4096, out_dim=6144]
            q_proj_grad = qkv_grad[:, :qkv_grad.shape[1]//3].T  # 取前 1/3 作为 Q 部分
            return q_proj_grad


def get_matrices_expansions(true_grads, emb_size, device=torch.device("cuda:0"), model_type="gpt", n_layers=2, rank_cutoff=0, model_dtype=torch.float16, B=None, tol=None, lora=True):
    if B is None:
        max_rank = 0
        for i in range(5):
            layer_grad = get_layer_grads(model_type, i, true_grads, lora=lora)
            if model_type in ["gpt2-full"]:
                layer_grad = layer_grad.T
            if i == 0: print("grad shape", layer_grad.shape)
            if model_dtype in [torch.float16, torch.bfloat16]:
                B = np.linalg.matrix_rank(layer_grad.to(dtype=torch.float32).cpu(), tol=tol)
            else:
                B = np.linalg.matrix_rank(layer_grad.cpu(), tol=tol)

            if max_rank < B:
                max_rank = B
        B = max_rank

    B = min(B, emb_size - rank_cutoff)

    R_Qs = []

    for i in range(n_layers):
        grad_Q = get_layer_grads(model_type, i, true_grads, lora=lora)
        if model_type == "gpt2-full":
            grad_Q = grad_Q.T
        _, R_Q = get_layer_decomp(grad_Q, B=B, tol=tol, model_dtype=model_dtype)
        R_Q = R_Q.to(device)
        R_Qs.append(R_Q)
    return B, R_Qs

def get_layer_decomp(grad, B=None, tol=None, model_dtype=torch.float32):
    upcast = model_dtype in [torch.float16, torch.bfloat16]

    grad = grad.detach().cpu()
    if upcast:
        grad = grad.to(dtype=torch.float32)
    grad_np = grad.numpy()

    if B is None:
        B = np.linalg.matrix_rank(grad_np, tol=tol)

    U, S, Vh = torch.svd_lowrank(torch.tensor(grad_np, dtype=torch.float32), q=B, niter=10)

    R = Vh.T
    if upcast:
        R = R.half()

    return B, R.detach()

def remove_padding(tokenizer, ids, PAD_TOKEN, left=False):
    if left:
        for i in range(ids.shape[0]):
            if ids[i].item() != PAD_TOKEN:
                ids = ids[i:]
                break
    else:
        for i in range(ids.shape[0] - 1, -1, -1):
            if ids[i].item() != PAD_TOKEN:
                ids = ids[:i+1]
                break
    return tokenizer.decode(ids)

def check_if_in_span(R_K_norm, v, norm='l2'):
    R_K_norm = R_K_norm.to(torch.float32)
    v = v.to(torch.float32)
    v /= v.pow(2).sum(-1,keepdim=True).sqrt()
    proj = torch.einsum('ik,ij,...j->...k', R_K_norm, R_K_norm, v )
    out_of_span = proj - v
    if norm == 'l2':
        size = out_of_span.pow(2).sum(-1).sqrt()
    elif norm == 'l1':
        size = out_of_span.abs().mean(-1)

    return size


def get_top_B_in_span(R_K_norm, v, thresh, norm, batch_size=1, compute_type="b<d", has_rope=False, dp=False):

    if has_rope:
        size = check_if_in_span(R_K_norm, v, norm)
        bools = size < thresh
        which = torch.where(bools)
        _, idx = torch.sort(size[which])
        which_new = []
        if dp: idx = idx[:200*batch_size]
        for w in which:
            which_new.append(w[idx])
        which_new = tuple(which_new)
        return which_new
    else:
        if compute_type == "b<d":
            thresh = 0.05
        while True:
            size = check_if_in_span(R_K_norm, v, norm)  # size: some tensor of distances/similarities
            bools = size < thresh                       # bools: which are below threshold
            which = torch.where(bools)                 # indices where condition is met

            # Get the values and their corresponding indices, sort them
            values = size[which]
            sorted_values, sorted_idx = torch.sort(values)

            if compute_type == "b<d":
                # Limit to batch_size
                if batch_size == 1 or len(sorted_idx) >= batch_size:
                    if batch_size == 1: sorted_idx = sorted_idx[:batch_size]
                    else:
                        sorted_idx = sorted_idx[:int(batch_size*1.5)]
                    break
                else:
                    thresh += 0.001
            else:
                if len(sorted_idx) >= int(batch_size*1.5):
                    sorted_idx = sorted_idx[:int(batch_size*1.5)]
                    break
                else:
                    thresh += 0.005

        # Apply to original indices
        which_new = []
        for w in which:
            which_new.append(w[sorted_idx])
        which_new = tuple(which_new)

        if which_new[1].shape[0] == 0:
            print(torch.min(size))

        return which_new

def filter_l1(client_model: BaseModel, R_Qs, l1_span_thresh, dist_norm, max_len, compute_type, dp=False, p_initial_index=1, batch_size=1):
    res_pos, res_ids, res_types = [], [], []

    sentence_ends = []
    n_tokens = 0

    tokenizer = client_model.tokenizer

    recovered_sentence_mas_shape = R_Qs[0].shape[0]//batch_size if R_Qs[0].shape[0] < client_model.emb_size else R_Qs[0].shape[0]

    p_max_len = min(tokenizer.model_max_length, max_len, recovered_sentence_mas_shape)


    for p in range(p_initial_index, p_max_len+1):

        with torch.no_grad(): embeds = client_model.get_embeddings(p)

        _, res_ids_new = get_top_B_in_span(R_Qs[0], embeds, l1_span_thresh, dist_norm, batch_size, compute_type, has_rope=client_model.has_rope(), dp=dp)

        res_types_new = torch.zeros_like(res_ids_new)
        res_pos_new = torch.ones_like(res_ids_new) * p

        embeds = embeds.to("cpu")
        del embeds
        torch.cuda.empty_cache()

        res_types += [res_types_new.tolist()]
        ids = res_ids_new.tolist()
        if len(ids) == 0 or p > p_max_len:
            break
        while client_model.eos_token in ids:
            end_token_ind = ids.index(client_model.eos_token)
            sentence_token_type = res_types[-1][end_token_ind]
            sentence_ends.append((p, sentence_token_type))
            ids = ids[:end_token_ind] + ids[end_token_ind + 1:]
            res_types[-1] = res_types[-1][:end_token_ind] + res_types[-1][end_token_ind + 1:]
        res_ids += [ids]
        res_pos += res_pos_new.tolist()
        n_tokens += len(ids)
        p += 1
        if client_model.has_rope():
            break

    return res_pos, res_ids, res_types, sentence_ends
