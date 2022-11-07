import os
import numpy as np
from PIL import Image
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms


class ImageNetDataset(Dataset):
    def __init__(self, args, image_dir, split):
        super(ImageNetDataset).__init__()
        self.args = args
        self.image_dir = os.path.join(image_dir, split)
        self.transform = transforms.Compose([
            transforms.Resize(self.args.image_size),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])


        self.cat_list = sorted(os.listdir(self.image_dir))

        self.image_list = []

        for cat in self.cat_list:
            name_list = sorted(os.listdir(os.path.join(self.image_dir, cat)))
            self.image_list += [os.path.join(self.image_dir, cat, image_name)
                                for image_name in name_list]

        print('Total %d Data.' % len(self.image_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument("--imagenet_dir", type=str)
    parser.add_argument("--process_output_dir", type=str)
    args = parser.parse_args()

    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    val_dataset = ImageNetDataset(
        args, image_dir=os.path.join(args.imagenet_dir, "CLS-LOC"), split='val')

    test_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=True
    )

    os.makedirs(args.process_output_dir, exist_ok=True)
    for i, image in enumerate(test_loader):
        if i == 1000:
            break
        np.save(os.path.join(args.process_output_dir, "%d.npy" % (i+1)), image)
