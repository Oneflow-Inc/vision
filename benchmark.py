import os
import oneflow as flow
import oneflow.nn as nn
from oneflow.utils.data import DataLoader
from oneflow.utils.vision import transforms
from oneflow.utils.vision.transforms import InterpolationMode
from oneflow.utils.vision.datasets import ImageFolder
from tqdm import tqdm
import numpy as np
from functools import partial

IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]

class ImageNetDataLoader(DataLoader):
    def __init__(self, data_dir, split='train', image_size=224, batch_size=16, num_workers=8):

        if split == 'train':
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256, interpolation=2) if image_size == 224 else transforms.Resize(image_size, interpolation=2),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
            ])
        
        self.dataset = ImageFolder(root=os.path.join(data_dir, split), transform=transform)
        super(ImageNetDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=True if split == 'train' else False,
            num_workers=num_workers)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.transpose(-1, -2)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k / batch_size * 100.0)
    return res


def test(model, data_dir, pretrained_path=None, batch_size=32, img_size=224, num_workers=8):

    if pretrained_path:
        state_dict = flow.load(pretrained_path)
        model.load_state_dict(state_dict)
        print("Load pretrained weights from {}".format(pretrained_path))
    
    model.cuda()
    
    data_loader = ImageNetDataLoader(
        data_dir = data_dir,
        image_size = img_size,
        batch_size = batch_size,
        num_workers = num_workers,
        split='val'
    )
    total_batch = len(data_loader)

    print("Start Evaluation")
    acc1s = []
    acc5s = []
    model.eval()
    with flow.no_grad():
        pbar = tqdm(enumerate(data_loader), total=total_batch)
        for batch_idx, (data, target) in pbar:
            pbar.set_description("Batch {:05d}/{:05d}".format(batch_idx, total_batch))

            data = data.to("cuda")
            target = target.to("cuda")

            pred_logits = model(data)
            acc1, acc5 = accuracy(pred_logits, target, topk=(1, 5))

            acc1s.append(acc1.item())
            acc5s.append(acc5.item())

            pbar.set_postfix(acc1=acc1.item(), acc5=acc5.item())
        
    print("Evaluation on dataset {:s}, Acc@1: {:.4f}, Acc@5: {:.4f}".format("ImageNet", np.mean(acc1s), np.mean(acc5s)))
