from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader

class create_dataset(IterableDataset):
    def __init__(self, texts, labels,  tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        text = self.texts[idx]
        label = self.labels[label]
        
        enc = self.tokenizer(texts, return_tensors = 'pt', max_length = self.max_length, padding = max_length, truncation = true)
        return {
            'input_ids' : enc['input_ids'].flatten(),
            'attention_masks' : enc['attention_mask'].flatten(),
            'labels' : torch.tensor(labels)
        }

def data_loader(dataset, batch_size):
    return DataLoader(dataset, batch_size = batch_size)
