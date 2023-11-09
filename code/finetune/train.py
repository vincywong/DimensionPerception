import os
import argparse
import sys
import torch
import deepspeed
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import LlamaForCausalLM, LlamaTokenizer
import transformers
import pickle
import pdb
import json
from peft import LoraConfig

from dataset import GPT2Dataset_onlyres, DatasetIds
from utils import flash_attn_forward, flash_attn_prepare_decoder_attention_mask
from peft import (
    get_peft_model,
    PeftModel
)

import random

def replace_llama_attn_with_flash_attn():
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = flash_attn_prepare_decoder_attention_mask
    transformers.models.llama.modeling_llama.LlamaAttention.forward = flash_attn_forward


def get_model_layers(model):
    layers = [["", model]]
    i = 0
    while i < len(layers):
        for nc, lc in layers[i][1].named_children():
            layers.append([f"{layers[i][0]}.{nc}" if layers[i][0] else nc, lc])
        i += 1
    return layers

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", type=str, required=True)
    parser.add_argument("--local_rank",type=int,default=-1,help="local_rank for distributed training on gpus")
    parser.add_argument("--seed",type=int,default=10,help="random seed")
    parser.add_argument("--dataset_type",choices=['GPT2Dataset_onlyres','BertDataset_onlyres', 'DatasetIds'],help="The type of dataset for dataloader")

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    
    train_config = json.load(open(args.train_config, "r"))
    
    print(train_config['ds_config'])

    random.seed(args.seed)

    for path in [train_config['training']['save_dir'], os.path.join(train_config['training']['save_dir'], train_config['training']['save_name'])]:
        if not os.path.exists(path):
            os.mkdir(path)

    device = torch.device("cuda")
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
    deepspeed.init_distributed()
    
    tokenizer = LlamaTokenizer.from_pretrained(train_config['tokenizer']['model_name_or_path'])
    print('tokenizer:', train_config['tokenizer']['model_name_or_path'])
    
    # if not train_config['training']['use_lora']:
        # st = ["<end>"]
        # tokenizer.add_special_tokens({'additional_special_tokens': tokenizer.additional_special_tokens + st})
        # print(tokenizer.additional_special_tokens)
    print('additional_special_tokens:', tokenizer.additional_special_tokens)

    datas = pickle.load(open(train_config['dataset']['data_ids_path'], "rb"))

    train_dataset = DatasetIds(
        tokenizer,
        datas, # your data preprocessing function
        train_config['dataset']['max_length'] # your max input length
    )
    print('dataset loaded!')

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        sampler=train_sampler,
        batch_size=train_config['ds_config']["train_micro_batch_size_per_gpu"]
    )

    model = LlamaForCausalLM.from_pretrained(train_config['model']['model_name_or_path'], low_cpu_mem_usage=True)
    print('model:', train_config['model']['model_name_or_path'])
    
    if train_config['training']['use_flash_attention']:
        print('using flash attn!!')
        replace_llama_attn_with_flash_attn()
    else:
        print('not using flash attn!!')

    if not train_config['training']['use_lora'] and tokenizer.additional_special_tokens != []:
        model.resize_token_embeddings(len(tokenizer))
        model_layers = get_model_layers(model)
        for layer in model_layers:
            if layer[0] in ['model.embed_tokens']:
                begin_idx = tokenizer.convert_tokens_to_ids(tokenizer.additional_special_tokens[0])
                end_idx = begin_idx + len(tokenizer.additional_special_tokens)
                print('normalize special token...')
                print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(torch.tensor([begin_idx]))))
                # print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(torch.tensor([end_idx - 1]))))
                torch.nn.init.normal_(layer[1].weight.data[begin_idx:end_idx], std=1e-6)
    else:
        if 'lora' in train_config['model']:
            # load lora parameter
            print('parameter loaded!')
            # print(args.load_lora_path)
            model = PeftModel.from_pretrained(model, train_config['lora_path'], is_trainable= True)
        else:
            # training from scratch
            print('training from scratch')
            lora_config = LoraConfig(**train_config['lora_config'])
            model = get_peft_model(model, lora_config)

    engine, _, _, _ = deepspeed.initialize(
        config=train_config['ds_config'],
        model=model, 
        model_parameters=model.parameters(),
    )
    print("model loaded.")

    train_config['training']['max_steps'] = train_config['training']['max_epoches'] * len(train_dataloader)

    global_step = 0
    engine.train()
    for epoch in range(train_config['training']['max_epoches']):
        losses = []
        if torch.distributed.get_rank() != -1:
            train_sampler.set_epoch(epoch)
        if torch.distributed.get_rank() == 0:
            pbar = tqdm(range(len(train_dataloader)))

        for batch in train_dataloader:
            # pdb.set_trace()
            loss = engine(
                input_ids = batch[0].to(device),
                labels = batch[1].to(device),
                attention_mask = batch[2].to(device),
                use_cache=False
            ).loss

            engine.backward(loss)
            engine.step()

            global_step += 1
            losses.append(loss.item())
            if global_step % train_config['training']['save_steps'] == 0:
                dist.barrier()
                if torch.distributed.get_rank() == 0:
                    if train_config['training']['use_lora']:
                        model.save_pretrained(os.path.join(train_config["training"]["save_dir"], train_config["training"]["save_name"], "weights", str(global_step)))
                        # model.save_pretrained(f"{args.save_dir + args.save_name + '/' + args.save_name}_{global_step}")
                    else:
                        engine.save_pretrained(os.path.join(train_config["training"]["save_dir"], train_config["training"]["save_name"], "weights", str(global_step)))
                        # engine.save_pretrained(f"{args.save_dir + args.save_name + '/' + args.save_name}_{global_step}")
                    tokenizer.save_pretrained(os.path.join(train_config["training"]["save_dir"], train_config["training"]["save_name"], "weights", str(global_step)))
                dist.barrier()

            if torch.distributed.get_rank() == 0:
                pbar.update()
                pbar.set_description(f"loss: {sum(losses[-200: ]) / len(losses[-200: ])}")

            if global_step >= train_config['training']['max_steps']:
                break
        

        dist.barrier()
        if torch.distributed.get_rank() == 0:
            if train_config['training']['use_lora']:
                model.save_pretrained(os.path.join(train_config["training"]["save_dir"], train_config["training"]["save_name"], "weights", "ep_" + str(epoch)))
                # model.save_pretrained(f"{args.save_dir + args.save_name + '/' + args.save_name}_epoch{epoch}")
            else:
                engine.save_pretrained(os.path.join(train_config["training"]["save_dir"], train_config["training"]["save_name"], "weights", "ep_" + str(global_step)))
                # engine.save_pretrained(f"{args.save_dir + args.save_name + '/' + args.save_name}_epoch{epoch}")
            tokenizer.save_pretrained(os.path.join(train_config["training"]["save_dir"], train_config["training"]["save_name"], "weights", "ep_" + str(global_step)))
        dist.barrier()

        if torch.distributed.get_rank() == 0:
            pbar.close()
        if global_step >= train_config['training']['max_steps']:
            break