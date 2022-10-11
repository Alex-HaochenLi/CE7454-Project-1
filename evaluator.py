import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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


class Evaluator:
    def __init__(
        self,
        net: nn.Module,
        k: int,
    ):
        self.net = net
        self.k = k

    def eval_recall(
        self,
        data_loader: DataLoader,
    ):
        self.net.eval()
        loss_avg = 0.0
        pred_list, gt_list = [], []

        # generate prompts
        raw_text = [relation_dict[i] for i in relation_dict.keys()]
        text_tokens = clip.tokenize(raw_text).cuda()

        with torch.no_grad():
            for batch in data_loader:
                data = batch[0].cuda()
                target = batch[1]['soft_label'].cuda()
                # relation = batch[2]
                loss, prob = self.net(data, target, text_tokens)
                loss_avg += float(loss.item())
                # gather prediction and gt
                pred = torch.topk(prob.data, self.k)[1]
                pred = pred.cpu().detach().tolist()
                pred_list.extend(pred)
                for soft_label in batch[1]['soft_label']:
                    gt_label = (soft_label == 1).nonzero(as_tuple=True)[0]\
                                .cpu().detach().tolist()
                    gt_list.append(gt_label)

        # compute mean recall
        score_list = np.zeros([56, 2], dtype=int)
        for gt, pred in zip(gt_list, pred_list):
            for gt_id in gt:
                # pos 0 for counting all existing relations
                score_list[gt_id][0] += 1
                if gt_id in pred:
                    # pos 1 for counting relations that is recalled
                    score_list[gt_id][1] += 1
        score_list = score_list[6:]
        # to avoid nan
        score_list[:, 0][score_list[:, 0] == 0] = 1
        meanrecall = np.mean(score_list[:, 1] / score_list[:, 0])

        metrics = {}
        metrics['test_loss'] = loss_avg / len(data_loader)
        metrics['mean_recall'] = meanrecall

        return metrics

    def submit(
        self,
        data_loader: DataLoader,
    ):
        self.net.eval()

        pred_list = []
        raw_text = [(relation_dict[i]) for i in relation_dict.keys()]
        text_tokens = clip.tokenize(raw_text).cuda()
        with torch.no_grad():
            for batch in data_loader:
                data = batch[0].cuda()
                _, logits = self.net(data, text=text_tokens)
                prob = torch.sigmoid(logits)
                pred = torch.topk(prob.data, self.k)[1] + 6
                pred = pred.cpu().detach().tolist()
                pred_list.extend(pred)
        return pred_list
