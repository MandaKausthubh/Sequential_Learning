import torch
from PIL import Image
from torch.utils.data import Dataset

def clip_collate(batch):
    images = [b["image"] for b in batch]
    texts = [b["text"] for b in batch]
    labels = None
    if "label" in batch[0]:
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {"image": images, "text": texts, "labels": labels}

class MultiModalDataset(Dataset):
    """
    For CLIP tasks: returns image + text (+ label).
    """
    def __init__(self, items):
        """
        items: list of dicts:
             - "image": path or PIL.Image
             - "text": str
             - optional "label"
        """
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        itm = self.items[idx]

        img = itm["image"]
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")

        out = {"image": img, "text": itm["text"]}

        if "label" in itm:
            out["label"] = itm["label"]

        return out
