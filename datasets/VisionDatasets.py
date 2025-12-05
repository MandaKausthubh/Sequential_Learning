# This code was borrowed and adapted from https://github.com/mlfoundations/model-soups/
import os
import torch
from torch.utils.data import SubsetRandomSampler
import numpy as np

from datasets.common import ImageFolderWithPaths, SubsetSampler
from datasets.imagenet_classnames import get_classnames

from PIL import Image

from imagenetv2_pytorch import ImageNetV2Dataset




# ----------- Imagenet ------------
class ImageNet:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=32,
                 classnames='openai',
                 distributed=False):
        self.preprocess = preprocess
        self.location = location
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classnames = get_classnames(classnames)
        self.distributed = distributed

        self.populate_train()
        self.populate_test()

    def populate_train(self):
        traindir = os.path.join(self.location, self.name(), 'train')
        self.train_dataset = ImageFolderWithPaths(
            traindir,
            transform=self.preprocess,
            )
        sampler = self.get_train_sampler()
        self.sampler = sampler
        kwargs = {'shuffle' : True} if sampler is None else {}
        # print('kwargs is', kwargs)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            **kwargs,
        )

    def populate_test(self):
        self.test_dataset = self.get_test_dataset()
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=self.get_test_sampler()
        )

    def get_test_path(self):
        test_path = os.path.join(self.location, self.name(), 'val_in_folder')
        if not os.path.exists(test_path):
            test_path = os.path.join(self.location, self.name(), 'val')
        return test_path

    def get_train_sampler(self):
        return torch.utils.data.distributed.DistributedSampler(self.train_dataset) if self.distributed else None


    def get_test_sampler(self):
        return None

    def get_test_dataset(self):
        return ImageFolderWithPaths(self.get_test_path(), transform=self.preprocess)

    def name(self):
        return 'imagenet'

class ImageNetTrain(ImageNet):

    def get_test_dataset(self):
        pass

def project_logits(logits, class_sublist_mask, device):
    if isinstance(logits, list):
        return [project_logits(l, class_sublist_mask, device) for l in logits]
    if logits.size(1) > sum(class_sublist_mask):
        return logits[:, class_sublist_mask].to(device)
    else:
        return logits.to(device)

class ImageNetSubsample(ImageNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        class_sublist, self.class_sublist_mask = self.get_class_sublist_and_mask()
        self.classnames = [self.classnames[i] for i in class_sublist]

    def get_class_sublist_and_mask(self):
        raise NotImplementedError()

    def populate_train(self):
        pass

    def project_logits(self, logits, device):
        return project_logits(logits, self.class_sublist_mask, device)

class ImageNetSubsampleValClasses(ImageNet):
    def get_class_sublist_and_mask(self):
        raise NotImplementedError()

    def populate_train(self):
        pass
    
    def get_test_sampler(self):
        self.class_sublist, self.class_sublist_mask = self.get_class_sublist_and_mask()
        idx_subsample_list = [range(x * 50, (x + 1) * 50) for x in self.class_sublist]
        idx_subsample_list = sorted([item for sublist in idx_subsample_list for item in sublist])
        
        sampler = SubsetSampler(idx_subsample_list)
        return sampler

    def project_labels(self, labels, device):
        projected_labels = [self.class_sublist.index(int(label)) for label in labels]
        return torch.LongTensor(projected_labels).to(device)

    def project_logits(self, logits, device):
        return project_logits(logits, self.class_sublist_mask, device)


class ImageNet98p(ImageNet):

    def get_train_sampler(self):
        idx_file = 'imagenet_98_idxs.npy'
        assert os.path.exists(idx_file)
        #if os.path.exists(idx_file):
        with open(idx_file, 'rb') as f:
            idxs = np.load(f)
        # else:
        #     idxs = np.zeros(len(self.train_dataset.targets))
        #     target_array = np.array(self.train_dataset.targets)
        #     for c in range(1000):
        #         m = target_array == c
        #         n = len(idxs[m])
        #         arr = np.zeros(n)
        #         arr[:26] = 1
        #         np.random.shuffle(arr)
        #         idxs[m] = arr
        #     with open(idx_file, 'wb') as f:
        #         np.save(f, idxs)

        idxs = (1 - idxs).astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])

        return sampler


class ImageNet2p(ImageNet):

    def get_train_sampler(self):
        idx_file = 'imagenet_98_idxs.npy'
        assert os.path.exists(idx_file)
        with open(idx_file, 'rb') as f:
            idxs = np.load(f)

        idxs = idxs.astype('int')
        sampler = SubsetSampler(np.where(idxs)[0])
        return sampler

class ImageNet2pShuffled(ImageNet):

    def get_train_sampler(self):
        print('shuffling val set.')
        idx_file = 'imagenet_98_idxs.npy'
        assert os.path.exists(idx_file)
        with open(idx_file, 'rb') as f:
            idxs = np.load(f)

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
        return sampler





# ----------- Imagenet-A ------------
IMAGENET_A_CLASS_SUBLIST = [
    6, 11, 13, 15, 17, 22, 23, 27, 30, 37, 39, 42, 47, 50, 57, 70, 71, 76, 79, 89, 90, 94, 96, 97, 99, 105, 107,
    108, 110,
    113, 124, 125, 130, 132, 143, 144, 150, 151, 207, 234, 235, 254, 277, 283, 287, 291, 295, 298, 301, 306, 307,
    308, 309,
    310, 311, 313, 314, 315, 317, 319, 323, 324, 326, 327, 330, 334, 335, 336, 347, 361, 363, 372, 378, 386, 397,
    400, 401,
    402, 404, 407, 411, 416, 417, 420, 425, 428, 430, 437, 438, 445, 456, 457, 461, 462, 470, 472, 483, 486, 488,
    492, 496,
    514, 516, 528, 530, 539, 542, 543, 549, 552, 557, 561, 562, 569, 572, 573, 575, 579, 589, 606, 607, 609, 614,
    626, 627,
    640, 641, 642, 643, 658, 668, 677, 682, 684, 687, 701, 704, 719, 736, 746, 749, 752, 758, 763, 765, 768, 773,
    774, 776,
    779, 780, 786, 792, 797, 802, 803, 804, 813, 815, 820, 823, 831, 833, 835, 839, 845, 847, 850, 859, 862, 870,
    879, 880,
    888, 890, 897, 900, 907, 913, 924, 932, 933, 934, 937, 943, 945, 947, 951, 954, 956, 957, 959, 971, 972, 980,
    981, 984,
    986, 987, 988]
IMAGENET_A_CLASS_SUBLIST_MASK = [(i in IMAGENET_A_CLASS_SUBLIST) for i in range(1000)]


class ImageNetAValClasses(ImageNetSubsampleValClasses):
    def get_class_sublist_and_mask(self):
        return IMAGENET_A_CLASS_SUBLIST, IMAGENET_A_CLASS_SUBLIST_MASK


class ImageNetA(ImageNetSubsample):
    def get_class_sublist_and_mask(self):
        return IMAGENET_A_CLASS_SUBLIST, IMAGENET_A_CLASS_SUBLIST_MASK

    def get_test_path(self):
        return os.path.join(self.location, 'imagenet-a')







# ----------- Imagenet-R ------------
IMAGENET_R_CLASS_SUBLIST = [
    1, 2, 4, 6, 8, 9, 11, 13, 22, 23, 26, 29, 31, 39, 47, 63, 71, 76, 79, 84, 90, 94, 96, 97, 99, 100, 105, 107,
    113, 122,
    125, 130, 132, 144, 145, 147, 148, 150, 151, 155, 160, 161, 162, 163, 171, 172, 178, 187, 195, 199, 203,
    207, 208, 219,
    231, 232, 234, 235, 242, 245, 247, 250, 251, 254, 259, 260, 263, 265, 267, 269, 276, 277, 281, 288, 289,
    291, 292, 293,
    296, 299, 301, 308, 309, 310, 311, 314, 315, 319, 323, 327, 330, 334, 335, 337, 338, 340, 341, 344, 347,
    353, 355, 361,
    362, 365, 366, 367, 368, 372, 388, 390, 393, 397, 401, 407, 413, 414, 425, 428, 430, 435, 437, 441, 447,
    448, 457, 462,
    463, 469, 470, 471, 472, 476, 483, 487, 515, 546, 555, 558, 570, 579, 583, 587, 593, 594, 596, 609, 613,
    617, 621, 629,
    637, 657, 658, 701, 717, 724, 763, 768, 774, 776, 779, 780, 787, 805, 812, 815, 820, 824, 833, 847, 852,
    866, 875, 883,
    889, 895, 907, 928, 931, 932, 933, 934, 936, 937, 943, 945, 947, 948, 949, 951, 953, 954, 957, 963, 965,
    967, 980, 981,
    983, 988]
IMAGENET_R_CLASS_SUBLIST_MASK = [(i in IMAGENET_R_CLASS_SUBLIST) for i in range(1000)]


class ImageNetRValClasses(ImageNetSubsampleValClasses):
    def get_class_sublist_and_mask(self):
        return IMAGENET_R_CLASS_SUBLIST, IMAGENET_R_CLASS_SUBLIST_MASK

class ImageNetR(ImageNetSubsample):
    def get_class_sublist_and_mask(self):
        return IMAGENET_R_CLASS_SUBLIST, IMAGENET_R_CLASS_SUBLIST_MASK

    def get_test_path(self):
        return os.path.join(self.location, 'imagenet-r')






# ----------- Imagenet-V2 ------------
class ImageNetV2DatasetWithPaths(ImageNetV2Dataset):
    def __getitem__(self, i):
        img, label = Image.open(self.fnames[i]), int(self.fnames[i].parent.name)
        if self.transform is not None:
            img = self.transform(img)
        return {
            'images': img,
            'labels': label,
            'image_paths': str(self.fnames[i])
        }

class ImageNetV2(ImageNet):
    def get_test_dataset(self):
        return ImageNetV2DatasetWithPaths(transform=self.preprocess, location=self.location)









# ----------- Imagenet-Sketch ------------
class ImageNetSketch(ImageNet):

    def populate_train(self):
        pass

    def get_test_path(self):
        return os.path.join(self.location, 'sketch')









