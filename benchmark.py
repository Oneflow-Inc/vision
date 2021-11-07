import os
import oneflow as flow
import oneflow.nn as nn
from oneflow.utils.data import DataLoader
from oneflow.utils.vision import transforms
from oneflow.utils.vision.transforms import InterpolationMode
from oneflow.utils.vision.datasets import ImageFolder
from torch.nn.functional import interpolate
from tqdm import tqdm
import numpy as np
from functools import partial
from flowvision.models import ModelCreator
import argparse

"""Model Specific Test
Swin-T: using interpolation "bicubic" for testing, which corresponds to interpolation=3 in Resize function
ViT: use mean=[0.5, 0.5, 0.5] and std=[0.5, 0.5, 0.5] for testing
"""

IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]

VIT_DEFAULT_MEAN = [0.5, 0.5, 0.5]
VIT_DEFAULT_STD = [0.5, 0.5, 0.5]


class ImageNetDataLoader(DataLoader):
    def __init__(
        self, data_dir, split="train", image_size=224, batch_size=16, num_workers=8
    ):

        if split == "train":
            transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(256, interpolation=3)  # 3: bibubic
                    if image_size == 224
                    else transforms.Resize(image_size, interpolation=3),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                ]
            )

        self.dataset = ImageFolder(
            root=os.path.join(data_dir, split), transform=transform
        )
        super(ImageNetDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=True if split == "train" else False,
            num_workers=num_workers,
        )


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.transpose(-1, -2)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def main(args):
    model = ModelCreator.create_model(args.model, pretrained=False)
    model.load_state_dict(flow.load('/home/kaijie/Documents/code/of/vision_git/vision/flowvision/models/ghostnet.zip'))
    data_dir = args.data_path
    img_size = args.img_size
    batch_size = args.batch_size
    num_workers = args.num_workers

    model.cuda()

    data_loader = ImageNetDataLoader(
        data_dir=data_dir,
        image_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        split="val",
    )
    total_batch = len(data_loader)

    print("Start Evaluation")
    Top_1_m = AverageMeter()
    Top_5_m = AverageMeter()
    model.eval()
    with flow.no_grad():
        pbar = tqdm(enumerate(data_loader), total=total_batch)
        for batch_idx, (data, target) in pbar:
            pbar.set_description("Batch {:05d}/{:05d}".format(batch_idx, total_batch))

            data = data.to("cuda")
            target = target.to("cuda")
            if args.model == "inception_v3":
                pred_logits, aux = model(data)
            elif args.model == "googlenet":
                pred_logits, aux1, aux2 = model(data)
            else:
                pred_logits = model(data)
            acc1, acc5 = accuracy(pred_logits, target, topk=(1, 5))

            Top_1_m.update(acc1.item(), pred_logits.size(0))
            Top_5_m.update(acc5.item(), pred_logits.size(0))

            pbar.set_postfix(acc1=acc1.item(), acc5=acc5.item())

    print(
        "Evaluation on dataset {:s}, Acc@1: {:.4f}, Acc@5: {:.4f}".format(
            "ImageNet", Top_1_m.avg, Top_5_m.avg
        )
    )


def _parse_args():
    parser = argparse.ArgumentParser("flags for benchmark test")
    parser.add_argument(
        "--model", type=str, required=True, help="model arch for test",
    )
    parser.add_argument(
        "--data_path", type=str, default="./", help="path to imagenet2012"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="test batch size")
    parser.add_argument("--img_size", type=int, default=224, help="test batch size")
    parser.add_argument(
        "--num_workers", type=int, default=8, help="num workers in dataloader"
    )
    return parser.parse_args()


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn")
    args = _parse_args()
    main(args)