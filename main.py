import argparse
import os
import time
import random
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from evaluator import Evaluator
from torch.utils.data import DataLoader
from torchvision.models import resnet50, resnet152
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101
from trainer import BaseTrainer
from model import Model


import clip
from dataset import PSGClsDataset


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='clip-vit')
parser.add_argument('--epoch', type=int, default=40)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--seed', type=int, default=1234)

args = parser.parse_args()
set_seed(args.seed)
savename = f'{args.model_name}'
os.makedirs('./checkpoints', exist_ok=True)
os.makedirs('./results', exist_ok=True)

# loading dataset
train_dataset = PSGClsDataset(stage='train')
train_dataloader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=8)

val_dataset = PSGClsDataset(stage='val')
val_dataloader = DataLoader(val_dataset,
                            batch_size=32,
                            shuffle=False,
                            num_workers=8)

test_dataset = PSGClsDataset(stage='test')
test_dataloader = DataLoader(test_dataset,
                             batch_size=32,
                             shuffle=False,
                             num_workers=8)
print('Data Loaded...', flush=True)

# loading model
# encoder = resnet152(pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder, _ = clip.load("ViT-B/32", device=device)
encoder.float()
model = Model(encoder)
model.cuda()
print('Model Loaded...', flush=True)

# loading trainer
trainer = BaseTrainer(model,
                      train_dataloader,
                      train_dataset,
                      learning_rate=args.lr,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay,
                      epochs=args.epoch)
evaluator = Evaluator(model, k=3)

# train!
print('Start Training...', flush=True)
begin_epoch = time.time()
best_val_recall = 0.0
for epoch in range(0, args.epoch):
    train_metrics = trainer.train_epoch()
    val_metrics = evaluator.eval_recall(val_dataloader)

    # show log
    print(
        '{} | Epoch {:3d} | Time {:5d}s | Train Loss {:.4f} | Test Loss {:.3f} | mR {:.2f}'
        .format(savename, (epoch + 1), int(time.time() - begin_epoch),
                train_metrics['train_loss'], val_metrics['test_loss'],
                100.0 * val_metrics['mean_recall']),
        flush=True)

    # save model
    if val_metrics['mean_recall'] >= best_val_recall:
        torch.save(model.state_dict(), f'./checkpoints/{savename}_best.ckpt')
        best_val_recall = val_metrics['mean_recall']

print('Training Completed...', flush=True)

# saving result!
print('Loading Best Ckpt...', flush=True)
checkpoint = torch.load(f'checkpoints/{savename}_best.ckpt')
model.load_state_dict(checkpoint)
test_evaluator = Evaluator(model, k=3)
check_metrics = test_evaluator.eval_recall(val_dataloader)
if best_val_recall == check_metrics['mean_recall']:
    print('Successfully load best checkpoint with acc {:.2f}'.format(
        100 * best_val_recall),
          flush=True)
else:
    print('Fail to load best checkpoint')
result = test_evaluator.submit(test_dataloader)


# save into the file
with open(f'results/{savename}_{best_val_recall}.txt', 'w') as writer:
    for label_list in result:
        a = [str(x) for x in label_list]
        save_str = ' '.join(a)
        writer.writelines(save_str + '\n')
print('Result Saved!', flush=True)
