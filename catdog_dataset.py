from torch.utils.data import Dataset
import torch
import os
from glob import glob
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

class CatDogClassification(Dataset):
    def __init__(self, dataset_dir, classes=['cat', 'dog'], transforms=None):
        # image_paths = glob(os.path.join(dataset_dir, classes[0], "*.jpg"))
        self.classes = classes
        self.transforms = transforms
        self.image_paths = []
        for c in classes:
            self.image_paths += glob(os.path.join(dataset_dir, c, "*.jpg"))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # image = cv2.imread(self.image_paths[idx])#, cv2.IMREAD_GRAYSCALE)
        # image = cv2.resize(image, (128,128))
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.get_label(self.image_paths[idx])
        if self.transforms:
            image = self.transforms(image)
            label = torch.tensor(label)
        return image, label

    def get_label(self, image_path):
        parent = Path(image_path).parts[-2]
        if parent=='cat':
            # label = np.array([0,1], dtype=np.float32)
            label = 0
        elif parent=='dog':
            # label = np.array([1,0], dtype=np.float32)
            label = 1
        else:
            print("Not registerd class detected!")
            label = None
        return label

# dataset = CatDogClassification(dataset_dir=os.path.join('dataset', 'test'), classes=['cat', 'dog'])
# # for idx in range(len(dataset)):
# idx = 0
# shuffled_idx = np.arange(len(dataset))
# np.random.shuffle(shuffled_idx)
# while True:
#     image, label = dataset[shuffled_idx[idx]]
#     print(f"{idx}/{len(dataset)-1}\tlabel: {label}")
#     cv2.imshow('image', image)
#     key = cv2.waitKey()

#     if key==ord('q'):
#         break
#     elif key==ord('a'):
#         idx -= 1
#     elif key==ord('d'):
#         idx += 1
#     if idx < 0:
#         idx = 0
#     if idx > len(dataset)-1:
#         idx = len(dataset)
    
