import numpy as np
from torch.utils.data import Dataset
import cv2
import os
from PIL import Image
from data.imgaug import GetTransforms
from data.utils import transform
from torchvision import transforms as tfs
import pandas as pd
import pdb
from tqdm import tqdm

np.random.seed(0)


class ImageDataset(Dataset):
    def __init__(self, label_path, cfg, mode='train', pt_transform=None,
                 sample_n=200):
        self.cfg = cfg
        self._label_header = ['pneumothorax']
        self._image_paths = []
        self._labels = []
        self._mode = mode
        self._transform = pt_transform
        self.dict = [{'1.0': '1', '': '0', '0.0': '0', '-1.0': '0'}]
        self.df = pd.read_csv(label_path).sample(sample_n)

        for i, row in tqdm(self.df.iterrows()):
            labels = []
            path = row['Path']
            label = str(row['Pneumothorax'])

            labels.append(self.dict[0].get(label))
            flg_enhance = True
            labels = list(map(int, labels))
            self._image_paths.append(path)
            assert os.path.exists(path), path
            self._labels.append(labels)
            if flg_enhance and self._mode == 'train':
                for i in range(self.cfg.enhance_times):
                    self._image_paths.append(path)
                    self._labels.append(labels)

        self._num_image = len(self._image_paths)

    def __len__(self):
        return self._num_image

    def __getitem__(self, idx):
        image = cv2.imread(self._image_paths[idx], 0)
        # image = np.array(Image.open(self._image_paths[idx]))
        # image = Image.fromarray(image)
        # for CXP
        # image = (((image-0.5720085)/0.32084802)*0.25411507)+0.5057236
        # for CXR
        # image = (((image - 0.5493168) / 0.34287995) * 0.25411507) + 0.5057236
        tmp_vals = np.load('nih_quantiles.npy', allow_pickle=True)
        src_values, src_unique_indices, src_counts = np.unique(image.ravel(),
                                                               return_inverse=True,
                                                               return_counts=True)
        src_quantiles = np.cumsum(src_counts) / image.size
        imgs = []
        for vals in tmp_vals:
            interp_a_values = np.interp(src_quantiles, vals[0], vals[1])
            imgs.append(interp_a_values[src_unique_indices].reshape(image.shape))
        image = np.mean(np.stack(imgs), axis=0)
        image = np.float32(image)
        image = transform(image, self.cfg)

        if self._mode == 'train':
            img_aug = tfs.Compose([
                tfs.ToTensor(),
                tfs.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05),
                                 scale=(0.95, 1.05), fillcolor=128),
                tfs.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
            ])
            image = img_aug(image)
            # image = GetTransforms(image, type=self.cfg.use_transforms_type)
        else:

            img_aug = tfs.Compose([
                tfs.ToTensor(),
                tfs.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
            ])
            image = img_aug(image)
        #image = image.float()

        labels = np.array(self._labels[idx]).astype(np.float32)

        path = self._image_paths[idx]
        if self._mode == 'train' or self._mode == 'valid':
            return (image, path, labels)
        elif self._mode == 'test':
            return (image, path)
        elif self._mode == 'heatmap':
            return (image, path, labels)
        else:
            raise Exception('Unknown mode : {}'.format(self._mode))
