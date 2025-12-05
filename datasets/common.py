from typing import Any, Dict, Tuple
import torchvision.datasets as datasets
from torch.utils.data import Sampler


class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)

class ImageFolderWithPaths(datasets.ImageFolder):
    def __init__(self, path, transform):
        super().__init__(path, transform)

    # pyright: ignore[override]
    def __getitem__(self, index:int) -> Dict[Any, Any]:
        image, label = super(ImageFolderWithPaths, self).__getitem__(index)
        return {
            'images': image,
            'labels': label,
            'image_paths': self.samples[index][0]
        }
