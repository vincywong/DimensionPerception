import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class GPT2Dataset_onlyres(Dataset):
    '''
    Dataset construction for training GPT-2 model, without padding. 
    Truncation is done using the end-of-sequence (EOS) token, 
    and only the loss for the response is computed.
    '''
    def __init__(self, tokenizer, datas, max_length):
        super().__init__()
        self.datas = datas
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.index = 0
        
        if not self.tokenizer.bos_token:
            self.tokenizer.bos_token = "<s>"
        if not self.tokenizer.eos_token:
            self.tokenizer.eos_token = "</s>"
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._preprocess()
    
    def _preprocess(self):
        self.input_ids = []

        self.labels = []
        
        for data in tqdm(self.datas):
            sample_input_ids = []
            sample_labels = []

            for idx, item in enumerate(data):
                    
                input, output = item['instruction'] + item['input'], item['output']

                input_tokens = self.tokenizer(input, padding=False, truncation=False, add_special_tokens=False)
                input_tokens = input_tokens["input_ids"][:self.max_length // 3]

                input_len = len(input_tokens)
                output_tokens = self.tokenizer(output, padding=False, truncation=False, add_special_tokens=False)
                output_tokens = output_tokens["input_ids"][:2 * (self.max_length // 3) - 1]

                sample_input_ids = input_tokens + output_tokens
                sample_labels = [-100] * input_len + output_tokens
                
                break
            
            self.input_ids += sample_input_ids
            self.labels += sample_labels
        
            self.input_ids += [self.tokenizer.eos_token_id]
            self.labels += [self.tokenizer.eos_token_id]   

        self.attention_mask = [1] * len(self.input_ids)
    
    def __len__(self):
        return (len(self.input_ids) - 1) // self.max_length + 1
    
    def __getitem__(self, index):
        return torch.tensor(self.input_ids[index * self.max_length : (index + 1) * self.max_length]), \
                torch.tensor(self.labels[index * self.max_length : (index + 1) * self.max_length]), \
                    torch.tensor(self.attention_mask[index * self.max_length : (index + 1) * self.max_length])

class DatasetIds(Dataset):
    def __init__(self, tokenizer, datas, max_length, **kwargs):
        super().__init__()
        self.input_ids = datas['input_ids']
        self.attention_mask = datas['attention_mask']
        self.labels = datas['labels']
        self.max_length = max_length

    def __len__(self):
        return len(self.input_ids) // self.max_length

    def __getitem__(self, index):
        return torch.tensor(self.input_ids[index * self.max_length : (index + 1) * self.max_length]), \
                torch.tensor(self.labels[index * self.max_length : (index + 1) * self.max_length]), \
                    torch.tensor(self.attention_mask[index * self.max_length : (index + 1) * self.max_length])
