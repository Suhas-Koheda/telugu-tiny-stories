import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from model.BPE import BPE

def load_dataset_():
    telugu_stories=load_dataset("neuralnets/multilingual-tinystories", split="te")
    return telugu_stories

class Dataset(Dataset):
    def __init__(self,txt,tokenizer,max_len,stride):
        self.input_ids=[]
        self.target_ids=[]
        token_ids=tokenizer.encode(txt)
        for i in range(0,len(token_ids)-max_len,stride):
            input_chunk=token_ids[i:i+max_len]
            target_chunk=token_ids[i+1:i+max_len+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self,idx):
        return self.input_ids[idx],self.target_ids[idx]

def create_loader(txt,tokenizer,batch_size=4,max_length=256,stride=128,shuffle=True,drop_last=True):
    dataset=Dataset(txt,tokenizer,max_length,stride)
    torch.save(
    {
        "inputs": dataset.input_ids,
        "targets": dataset.target_ids
    },
    "train_data.pt"
)
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,drop_last=drop_last)
    return dataloader

