from transformers import LlamaTokenizer
import argparse
import os
import json
import pickle
from dataset import GPT2Dataset_onlyres

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", type=str, required=True, default="./config/ft_dimperc.json")
    args = parser.parse_args()
    
    train_config = json.load(open(args.train_config, "r"))
    
    if not os.path.exists(train_config['dataset']['data_ids_path']):
        tokenizer = LlamaTokenizer.from_pretrained(train_config['tokenizer']['model_name_or_path'])
        with open(train_config['dataset']['data_path'], "rb") as f:
            datas = pickle.load(f)
        
        train_dataset = GPT2Dataset_onlyres(tokenizer, datas, train_config['dataset']['max_length'])
        # datas = get_multiround_data(train_config['dataset']['setup_arguments']['data_path'], 0)
        
        pickle.dump(
            {
                "input_ids": train_dataset.input_ids,
                "labels": train_dataset.labels,
                "attention_mask": train_dataset.attention_mask
            },
            open(train_config['dataset']['data_ids_path'], "wb")
        )
        
        example_tokens = [tokenizer.decode(ids) for ids in train_dataset.input_ids[:400]]
        
    else:
        print("The data ids file already exists")