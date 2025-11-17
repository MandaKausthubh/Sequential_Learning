import torch
from torch.utils.data import Dataset

def text_collate(batch):
    texts = [b["text"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {"text": texts, "labels": labels}

class TextDataset(Dataset):
    def __init__(self, items):
        """
        items: list of dicts with keys:
            - "text": str
            - "label": int
        """
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        itm = self.items[idx]
        return {"text": itm["text"], "label": itm["label"]}
