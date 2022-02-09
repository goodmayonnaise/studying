from torch.utils.data import Dataset
import pandas as pd
import torch
from tokenizer import Tokenizer
from config import Config

class NaverDataSet(Dataset):
    def __init__(self, data_path, tokenizer, max_len):
        super().__init__()
        self.data_path = data_path
        self.tok = tokenizer
        self.max_len = max_len
        self.data = pd.read_csv(self.data_path, sep='\t', header=0)
        self.data.dropna(axis=0, inplace=True)

    def __getitem__(self, index):
        temp_data = self.data.iloc[index]
        sentence, label = temp_data.document, temp_data.label
        input_id = self.tok.encodeAsids(sentence)

        if len(input_id) >= self.max_len:
            input_id = input_id[:self.max_len]

        else:
            input_id =  input_id.tolist() + [self.tok.pad_id()] * (self.max_len - len(input_id))

        return torch.LongTensor(input_id), torch.LongTensor([label])


    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    
    config = Config("/Users/khs/Desktop/김현수/studying/projects/SentimentClassification/hyunsoo/config.json")
    tokenizer = Tokenizer(config)

    nds = NaverDataSet(config.train_file, tokenizer, config.max_len)
    print(nds[0])
    print(len(nds))