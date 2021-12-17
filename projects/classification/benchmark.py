import os
import oneflow as flow
from oneflow.utils.data import DataLoader
from oneflow.utils.vision import transforms
from oneflow.utils.vision.transforms import InterpolationMode
from oneflow.utils.vision.datasets import ImageFolder
from tqdm import tqdm
from flowvision.models import ModelCreator
from flowvision.transforms.functional import str_to_interp_mode
import argparse
import math
import json

"""Model Specific Test
Swin-T: using interpolation "bicubic" for testing, which corresponds to interpolation=3 in Resize function
ViT: use mean=[0.5, 0.5, 0.5] and std=[0.5, 0.5, 0.5] for testing
CSWin: using DEFAULT_CROP_SIZE = 0.9
"""


def get_mean_std(mode="imagenet_default_mean_std"):
    if mode == "imagenet_default_mean_std":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif mode == "vit_mean_std":
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    else:
        raise NotImplementedError(f"Unkown mode: {mode}")
    return mean, std


class ImageNetDataLoader(DataLoader):
    def __init__(
        self,
        data_dir,
        split="train",
        image_size=224,
        img_mean=(0.485, 0.456, 0.406),
        img_std=(0.229, 0.224, 0.225),
        crop_pct=0.875,
        interpolation="bibubic",
        batch_size=16,
        num_workers=8,
    ):

        if split == "train":
            transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(img_mean, img_std),
                ]
            )
        else:
            scale_size = int(math.floor(image_size / crop_pct))
            transform = transforms.Compose(
                [
                    transforms.Resize(
                        scale_size, interpolation=str_to_interp_mode(interpolation)
                    )  # 3: bibubic
                    if image_size == 224
                    else transforms.Resize(image_size, interpolation=3),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(img_mean, img_std),
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

    return [correct[:k].reshape(-1).float().sum(0) * 100.0 / batch_size for k in topk]

def accuracy_r(output, names, real_reables, topk=(1,)):
    '''

    Args:
        output:
        names: batch images names
        real_reables: list: sorted by images names
        topk:

    Returns:

    '''
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    ret = []
    for k in topk:
        batch_size = len(names)
        correct = 0
        for i, name in enumerate(names):
            if real_reables[int(name)-1]:
                real_targets = real_reables[int(name)-1]
                ps = pred[i][:k]
                for j in ps:
                    if int(j.item()) in real_targets:
                        correct += 1
                        break
            else: # dont care this image, so we decrease batch_size
                batch_size -= 1
        ret.append(correct/batch_size*100)
        ret.append(batch_size)
    return ret


def main(args):
    model = ModelCreator.create_model(args.model, pretrained=True)
    data_dir = args.data_path
    img_size = args.img_size
    img_mean, img_std = get_mean_std(args.normalize_mode)
    crop_pct = args.crop_pct
    interpolation = args.interpolation
    batch_size = args.batch_size
    num_workers = args.num_workers

    model.cuda()

    data_loader = ImageNetDataLoader(
        data_dir=data_dir,
        image_size=img_size,
        img_mean=img_mean,
        img_std=img_std,
        crop_pct=crop_pct,
        interpolation=interpolation,
        batch_size=batch_size,
        num_workers=num_workers,
        split="val",
    )
    total_batch = len(data_loader)


    print("Start Evaluation")
    Top_1_m = AverageMeter()
    Top_5_m = AverageMeter()
    if args.real_labels:
        with open('real.json') as f:
            real_labels = json.load(f)
        names = [name[-11:-5] for name, cls in data_loader.sampler.data_source.imgs]
        Top_1r_m = AverageMeter()
        Top_5r_m = AverageMeter()
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
            if args.real_labels:
                acc1_r, bs1, acc5_r, bs2 = accuracy_r(pred_logits, names[args.batch_size*batch_idx: args.batch_size*batch_idx + len(target)], real_labels, topk=(1, 5))
                Top_1r_m.update(acc1_r, bs1)
                Top_5r_m.update(acc5_r, bs2)
                pbar.set_postfix(acc1=acc1.item(), acc5=acc5.item(), acc1_r=acc1_r, acc5_r=acc5_r)
            else:
                pbar.set_postfix(acc1=acc1.item(), acc5=acc5.item(),)

    if args.real_labels:
        print(
        "Evaluation {:s} on dataset {:s}, Acc@1: {:.4f}, Acc@5: {:.4f}, Acc@1_r: {:.4f}, Acc@5_r: {:.4f}".format(
            args.model, "ImageNet", Top_1_m.avg, Top_5_m.avg, Top_1r_m.avg, Top_5r_m.avg
        )
    )
    else:
        print(
            "Evaluation {:s} on dataset {:s}, Acc@1: {:.4f}, Acc@5: {:.4f}, ".format(
                args.model, "ImageNet", Top_1_m.avg, Top_5_m.avg,
            ))



def _parse_args():
    parser = argparse.ArgumentParser("flags for benchmark test")
    parser.add_argument(
        "--model", type=str, required=True, help="model arch for test",
    )
    parser.add_argument(
        "--data_path", type=str, default="./", help="path to imagenet2012"
    )
    parser.add_argument(
        "--real_labels", action='store_true', help="where evaluate real label"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="test batch size")
    parser.add_argument("--img_size", type=int, default=224, help="test batch size")
    parser.add_argument(
        "--num_workers", type=int, default=8, help="num workers in dataloader"
    )
    parser.add_argument(
        "--normalize_mode",
        type=str,
        default="imagenet_default_mean_std",
        choices=["imagenet_default_mean_std", "vit_mean_std"],
        help="the normalization mode",
    )
    parser.add_argument(
        "--crop_pct", type=float, default=0.875, help="image crop ratio controller"
    )
    parser.add_argument(
        "--interpolation",
        type=str,
        default="bicubic",
        help="interpolation method to choose",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args)
