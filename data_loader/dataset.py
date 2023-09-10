from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch

class create_dataset(Dataset):
    def __init__(self, texts, labels,  tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        text = self.texts[idx]
        label = self.labels[idx]
        
        enc = self.tokenizer(text, return_tensors = 'pt', max_length = self.max_length, padding = 'max_length', truncation = True)
        return {
            'input_ids' : enc['input_ids'].flatten(),
            'attention_masks' : enc['attention_mask'].flatten(),
            'labels' : torch.tensor(label)
        }        

def data_loader(dataset, batch_size):
    return DataLoader(dataset=dataset, batch_size = batch_size)
