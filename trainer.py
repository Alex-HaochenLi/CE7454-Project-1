import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import clip

relation_dict = {0: 'hanging from',
                 1: 'on back of', 2: 'falling of', 3: 'going down', 4: 'painted on', 5: 'waling on', 6: 'running on',
                 7: 'crossing', 8: 'standing on', 9: 'lying on', 10: 'sitting on', 11: 'flying over', 12: 'jumping over',
                 13: 'jumping from', 14: 'wearing', 15: 'holding', 16: 'carrying', 17: 'looking at', 18: 'guiding',
                 19: 'kissing', 20: 'eating', 21: 'drinking', 22: 'feeding', 23: 'biting', 24: 'catching', 25: 'picking',
                 26: 'playing with', 27: 'chasing', 28: 'climbing', 29: 'cleaning', 30: 'playing', 31: 'touching',
                 32: 'pushing', 33: 'pulling', 34: 'opening', 35: 'cooking', 36: 'talking to', 37: 'throwing',
                 38: 'slicing', 39: 'driving', 40: 'riding', 41: 'parked on', 42: 'driving on', 43: 'about to hit',
                 44: 'kicking', 45: 'swinging', 46: 'entering', 47: 'exiting', 48: 'enclosing', 49: 'learning on'}

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max -
                     lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


class BaseTrainer:
    def __init__(self,
                 net: nn.Module,
                 train_loader: DataLoader,
                 train_dataset: Dataset,
                 learning_rate: float = 0.1,
                 momentum: float = 0.9,
                 weight_decay: float = 0.0005,
                 epochs: int = 100) -> None:
        self.net = net
        self.train_loader = train_loader
        self.train_dataset = train_dataset

        self.optimizer = torch.optim.AdamW(
            [{'params': net.encoder.parameters(), 'lr': 1e-7}],
            learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                epochs * len(train_loader),
                1,  # since lr_lambda computes multiplicative factor
                1e-6 / learning_rate,
            ),
        )

    def train_epoch(self):
        self.net.train()  # enter train mode

        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

        # raw_text = [(relation_dict[i]) for i in relation_dict.keys()]
        # text_tokens = clip.tokenize(raw_text).cuda()

        for train_step in tqdm(range(1, len(train_dataiter) + 1)):
            # for train_step in tqdm(range(1, 5)):
            batch = next(train_dataiter)
            data = batch[0].cuda()
            target = batch[1]['soft_label'].cuda()
            relation = batch[2]

            raw_text = []
            for batch in relation:
                for item in batch:
                    if item[0] >= 0:
                        try:
                            raw_text.append(self.train_dataset.item_dict[item[0].item()] + ' ' + relation_dict[item[2].item()] + ' ' +
                                        self.train_dataset.item_dict[item[1].item()])
                        except KeyError:
                            print(1)
            text_tokens = clip.tokenize(raw_text).cuda()
            relation_num = (relation[:, :, 0] >= 0).sum(1).cuda()
            # forward
            loss, _ = self.net(data, target, text_tokens, relation_num)
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        metrics = {}
        metrics['train_loss'] = loss_avg

        return metrics
