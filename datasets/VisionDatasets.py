import os
from typing import Callable, Optional, Any, Dict
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets




class BaseVisionDataset(Dataset):
    """
    Consistent wrapper so that every dataset returns:
        { "image": PIL.Image, "label": int }
    """

    def __init__(self, root: str, transform: Optional[Callable] = None):
        super().__init__()
        self.root = root
        self.transform = transform

    def __getitem__(self, index) -> Dict[str, Any]:
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError



class ImageNet(BaseVisionDataset):
    """
    Standard ImageNet-1k dataset using torchvision structure.
    Expected directory structure:
        root/imagenet/train/<class>/*.JPEG
        root/imagenet/val/<class>/*.JPEG
    """

    def __init__(self, root: str, split: str = "train",
                 transform: Optional[Callable] = None):

        super().__init__(root, transform)
        assert split in ["train", "val"]

        split_folder = "train" if split == "train" else "val"
        self.ds = datasets.ImageFolder(
            root=os.path.join(root, split_folder),
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        img, label = self.ds[index]  # img is PIL already

        if self.transform:
            img = self.transform(img)

        return {"image": img, "label": label}




class ImageNetV2MF(BaseVisionDataset):
    """
    ImageNetV2 matched-frequency dataset.
    Expected structure (standard from dataset creators):
        root/ImageNetV2-matched-frequency/ (images)
        root/ImageNetV2-matched-frequency/val_labels.txt (labels 0-999)
    """

    def __init__(self, root: str,
                 transform: Optional[Callable] = None):

        super().__init__(root, transform)

        # All images directly in root, alphabetical order matched to labels
        self.img_files = sorted([
            f for f in os.listdir(root)
            if f.lower().endswith(("jpeg", "jpg", "png"))
        ])

        labels_file = os.path.join(root, "imagenetv2-matched-frequency-labels.txt")
        if not os.path.exists(labels_file):
            raise FileNotFoundError(
                f"Expected label file not found: {labels_file}"
            )

        # Load labels
        self.labels = []
        with open(labels_file, "r") as f:
            for line in f:
                self.labels.append(int(line.strip()))

        assert len(self.labels) == len(self.img_files), \
            "Mismatch: number of images vs labels"

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.img_files[index])
        img = Image.open(img_path).convert("RGB")
        label = self.labels[index]

        if self.transform:
            img = self.transform(img)

        return {"image": img, "label": label}





class ImageNetA(BaseVisionDataset):
    """
    ImageNet-A directory structure:
        root/ImageNet-A/<class_name>/*.png or *.jpg
    The folder names map to synset names.
    You must provide a mapping synset -> class index (1000-way).
    """

    def __init__(self, root: str,
                 class_map_path: str,
                 transform: Optional[Callable] = None):
        """
        class_map_path: path to imagenet_class_index.json from torchvision
        """
        super().__init__(root, transform)

        import json
        with open(class_map_path, "r") as f:
            cls_json = json.load(f)

        # Map synset -> 0..999
        self.syn_to_class = {v[0]: int(k) for k, v in cls_json.items()}

        self.samples = []
        for synset in sorted(os.listdir(root)):
            folder = os.path.join(root, synset)
            if not os.path.isdir(folder):
                continue

            if synset not in self.syn_to_class:
                continue

            target = self.syn_to_class[synset]

            for fname in os.listdir(folder):
                if fname.lower().endswith(("png", "jpg", "jpeg")):
                    self.samples.append((os.path.join(folder, fname), target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return {"image": img, "label": label}





class ImageNetR(BaseVisionDataset):
    """
    ImageNet-R directory structure:
        root/ImageNet-R/<synset>/*.png
    Expected: use same synset->class mapping as ImageNet-A.
    """

    def __init__(self, root: str,
                 class_map_path: str,
                 transform: Optional[Callable] = None):
        super().__init__(root, transform)

        import json
        with open(class_map_path, "r") as f:
            cls_json = json.load(f)

        self.syn_to_class = {v[0]: int(k) for k, v in cls_json.items()}

        self.samples = []
        for synset in sorted(os.listdir(root)):
            folder = os.path.join(root, synset)
            if not os.path.isdir(folder):
                continue

            if synset not in self.syn_to_class:
                continue

            target = self.syn_to_class[synset]

            for fname in os.listdir(folder):
                if fname.lower().endswith(("png", "jpg", "jpeg")):
                    self.samples.append((os.path.join(folder, fname), target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return {"image": img, "label": label}




class ImageNetSketch(BaseVisionDataset):
    """
    ImageNet-Sketch structure:
        root/sketch/<synset>/*.png
    Must use same synset -> class mapping file.
    """

    def __init__(self, root: str,
                 class_map_path: str,
                 transform: Optional[Callable] = None):
        super().__init__(root, transform)

        import json
        with open(class_map_path, "r") as f:
            cls_json = json.load(f)

        self.syn_to_class = {v[0]: int(k) for k, v in cls_json.items()}

        self.samples = []
        for synset in sorted(os.listdir(root)):
            folder = os.path.join(root, synset)
            if not os.path.isdir(folder):
                continue

            if synset not in self.syn_to_class:
                continue

            target = self.syn_to_class[synset]

            for fname in os.listdir(folder):
                if fname.lower().endswith(("png", "jpg", "jpeg")):
                    self.samples.append((os.path.join(folder, fname), target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return {"image": img, "label": label}
