import io
import json
import logging
import os
import numpy as np
import torch
import torchvision.transforms as trn
from PIL import Image, ImageFile
from torch.utils.data import Dataset

import clip
from collections import defaultdict
# to fix "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


def get_transforms(stage: str):
    mean, std = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
    if stage == 'train':
        return trn.Compose([
            Convert('RGB'),
            # trn.Resize((1333, 800)),
            trn.Resize((224, 224)),
            trn.RandomHorizontalFlip(),
            # trn.RandomCrop((1333, 800), padding=4),
            trn.RandomCrop((224, 224), padding=4),
            trn.ToTensor(),
            trn.Normalize(mean, std),
        ])

    elif stage in ['val', 'test']:
        return trn.Compose([
            Convert('RGB'),
            # trn.Resize((1333, 800)),
            trn.Resize((224, 224)),
            trn.ToTensor(),
            trn.Normalize(mean, std),
        ])


class PSGClsDataset(Dataset):
    def __init__(
        self,
        stage,
        root='./data/coco/',
        num_classes=50,
    ):
        super(PSGClsDataset, self).__init__()
        self.relation_length = 30
        with open('./data/psg/psg_cls_basic.json') as f:
            dataset = json.load(f)
        with open('./data/psg/psg_cls_advanced.json') as f:
            dataset_advanced = json.load(f)
        self.item_dict = {}
        for i in range(len(dataset_advanced['thing_classes'])):
            self.item_dict[i] = dataset_advanced['thing_classes'][i]
        _ = len(self.item_dict)
        for i in range(len(dataset_advanced['stuff_classes'])):
            self.item_dict[i+_] = dataset_advanced['stuff_classes'][i]
        self.full_relations = {}
        for item in dataset_advanced['data']:
            self.full_relations[item['image_id']] = item['relations']

        self.imglist = [
            d for d in dataset['data']
            if d['image_id'] in dataset[f'{stage}_image_ids']
        ]
        self.root = root
        self.transform_image = get_transforms(stage)
        self.num_classes = num_classes
        self.stage = stage

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        sample = self.imglist[index]
        path = os.path.join(self.root, sample['file_name'])
        try:
            with open(path, 'rb') as f:
                content = f.read()
                filebytes = content
                buff = io.BytesIO(filebytes)
                image = Image.open(buff).convert('RGB')
                image = self.transform_image(image)
        except Exception as e:
            logging.error('Error, cannot read [{}]'.format(path))
            raise e
        # Generate Soft Label
        soft_label = torch.Tensor(self.num_classes)
        soft_label.fill_(0)
        soft_label[[(i - 6) for i in sample['relations']]] = 1
        sample['soft_label'] = soft_label
        del sample['relations']

        if self.stage == 'train' or self.stage == 'val':
            relations = self.full_relations[sample['image_id']]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in relations:
                if r >= 6:
                    all_rel_sets[(o0, o1)].append(r - 6)
            gt_rels = [(k[0], k[1], np.random.choice(v))
                       for k, v in all_rel_sets.items()]
            assert len(gt_rels) <= self.relation_length
            for i in range(self.relation_length - len(gt_rels)):
                gt_rels.append([-1, -1, -1])
            relations = torch.tensor(gt_rels)
        elif self.stage == 'test':
            relations = None

        if self.stage == 'train' or self.stage == 'val':
            return image, sample, relations
        elif self.stage == 'test':
            return image, sample
