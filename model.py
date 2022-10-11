import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.embed_dim = 64
        self.encoder = encoder
        # self.fc_img = nn.Sequential(nn.Linear(512, 2048),
        #                         nn.ReLU(),
        #                         nn.Linear(2048, self.embed_dim * 50))
        # self.fc = nn.Sequential(nn.Linear(self.embed_dim * 50, 1024),
        #                          nn.ReLU(),
        #                          nn.Linear(1024, 50))

    def forward(self, image, label=None, text=None, num=None):
        image_features = self.encoder.encode_image(image)
        text_features = self.encoder.encode_text(text)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        probs = image_features @ text_features.t()

        # image_features = self.fc_img(image_features)
        # vecs = image_features.view(image_features.size(0), 50, self.embed_dim)
        # vecs = vecs / vecs.norm(dim=2, keepdim=True)

        # regression loss
        regr_loss = 0
        if label is not None:
            if num is not None:
                label_matrix = torch.zeros(probs.size(0), probs.size(1), device=probs.device)
                for i in range(1, num.size(0)):
                    num[i] = num[i] + num[i-1]
                for i in range(len(num)):
                    if i == 0:
                        label_matrix[i, :num[i]] = 1
                    else:
                        label_matrix[i, num[i-1]:num[i]] = 1
            else:
                label_matrix = label

            probs = torch.nn.functional.softmax(probs / 0.07, dim=1)
            regr_loss = - torch.mean((torch.log(probs + 1e-15) * label_matrix).sum(1) / label_matrix.sum(1))
            regr_loss += - torch.mean((torch.log(probs.T + 1e-15) * label_matrix.T).sum(1))

            # probs2 = self.fc(image_features)
            # regr_loss += 10 * F.binary_cross_entropy_with_logits(probs2, label, reduction='mean')

        # contrastive learning
        # if self.training:
        #     label_matrix = torch.relu(torch.matmul(label.T.unsqueeze(2), label.T.unsqueeze(1)) - torch.eye(image.size(0)).cuda())
        #     mask = (label_matrix.sum(-1).sum(-1) > 0)
        #     label_matrix = label_matrix[mask]
        #     vecs = vecs[:, mask, :].permute(1, 0, 2)
        #     logits = torch.matmul(vecs, vecs.permute(0, 2, 1))
        #     logits = logits[:, torch.eye(logits.size(1)).cuda() != 1].view(logits.size(0), image.size(0), image.size(0) - 1)
        #     label_matrix = label_matrix[:, torch.eye(logits.size(1)).cuda() != 1].view(logits.size(0), image.size(0), image.size(0) - 1)
        #     logits = torch.nn.functional.softmax(logits / 0.07, dim=2).view(logits.size(0) * logits.size(1), logits.size(2))
        #
        #     label_matrix = label_matrix.view(label_matrix.size(0) * label_matrix.size(1), label_matrix.size(2))
        #     mask2 = (label_matrix.sum(1) != 0)
        #     label_matrix = label_matrix[mask2]
        #     logits = logits[mask2]
        #     contra_loss = (torch.log(logits + 1e-15) * label_matrix).sum(1) / label_matrix.sum(1)
        #     contra_loss = - torch.mean(contra_loss)
        #
        #     loss = regr_loss + contra_loss
        # else:
        #     loss = regr_loss
        return regr_loss, probs

