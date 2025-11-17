import torch
from PIL import Image
from torch.utils.data import Dataset

def vision_collate(batch):
    images = [b["image"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {"image": images, "labels": labels}

class VisionDataset(Dataset):
    def __init__(self, items):
        """
        items: list of dicts with keys:
            - "image": path or PIL.Image
            - "label": int
        """
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        itm = self.items[idx]
        img = itm["image"]
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        label = itm["label"]
        return {"image": img, "label": label}
