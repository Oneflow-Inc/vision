import argparse
import datetime
import os
import time

import oneflow as flow
import oneflow.utils.data
import flowvision
import flowvision.models.detection
from flowvision.models import ModelCreator

from coco_utils import get_coco, get_coco_kp
from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from engine import train_one_epoch, evaluate

import presets
import utils


def get_dataset(name, image_set, transform, data_path):
    paths = {"coco": (data_path, get_coco, 91), "coco_kp": (data_path, get_coco_kp, 2)}
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform(train, data_augmentation):
    return (
        presets.DetectionPresetTrain(data_augmentation)
        if train
        else presets.DetectionPresetEval()
    )


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="OneFlow Detection Training", add_help=True
    )

    parser.add_argument("--data-path", default="/dataset/mscoco_2017/", help="dataset")
    parser.add_argument("--dataset", default="coco", help="dataset")
    parser.add_argument("--model", default="ssd300_vgg16", help="model")
    parser.add_argument("--device", default="cuda", help="device")
    parser.add_argument(
        "-b",
        "--batch-size",
        default=2,
        type=int,
        help="images per gpu, the total batch size is $NGPU x batch_size",
    )
    parser.add_argument(
        "--epochs",
        default=26,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--lr",
        default=0.02,
        type=float,
        help="initial learning rate, 0.02 is the default value for training "
        "on 8 gpus and 2 images_per_gpu",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--lr-scheduler",
        default="multisteplr",
        help="the lr scheduler (default: multisteplr)",
    )
    parser.add_argument(
        "--lr-step-size",
        default=8,
        type=int,
        help="decrease lr every step-size epochs (multisteplr scheduler only)",
    )
    parser.add_argument(
        "--lr-steps",
        default=[16, 22],
        nargs="+",
        type=int,
        help="decrease lr every step-size epochs (multisteplr scheduler only)",
    )
    parser.add_argument(
        "--lr-gamma",
        default=0.1,
        type=float,
        help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)",
    )
    parser.add_argument("--print-freq", default=20, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", help="path where to save")
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)
    parser.add_argument(
        "--trainable-backbone-layers",
        default=None,
        type=int,
        help="number of trainable layers of backbone",
    )
    parser.add_argument(
        "--data-augmentation",
        default="hflip",
        help="data augmentation policy (default: hflip)",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument(
        "--world-size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist-url", default="env://", help="url used to set up distributed training"
    )

    return parser


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = flow.device(args.device)

    # Data loading code
    print("Loading data")

    dataset, num_classes = get_dataset(
        args.dataset,
        "train",
        get_transform(True, args.data_augmentation),
        args.data_path,
    )
    dataset_test, _ = get_dataset(
        args.dataset,
        "val",
        get_transform(False, args.data_augmentation),
        args.data_path,
    )

    print("Creating data loaders")
    if args.distributed:
        train_sampler = flow.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = flow.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = flow.utils.data.RandomSampler(dataset)
        test_sampler = flow.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(
            dataset, k=args.aspect_ratio_group_factor
        )
        train_batch_sampler = GroupedBatchSampler(
            train_sampler, group_ids, args.batch_size
        )
    else:
        train_batch_sampler = flow.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True
        )

    data_loader = flow.utils.data.DataLoader(
        dataset,
        batch_sampler=train_batch_sampler,
        num_workers=args.workers,
        collate_fn=utils.collate_fn,
    )

    data_loader_test = flow.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        sampler=test_sampler,
        num_workers=args.workers,
        collate_fn=utils.collate_fn,
    )

    print("Creating model")
    kwargs = {"trainable_backbone_layers": args.trainable_backbone_layers}
    model = ModelCreator.create_model(args.model, pretrained=args.pretrained)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = flow.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = flow.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "multisteplr":
        lr_scheduler = flow.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_steps, gamma=args.lr_gamma
        )
    elif args.lr_scheduler == "cosineannealinglr":
        lr_scheduler = flow.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
    else:
        raise RuntimeError(
            "Invalid lr scheduler '{}'. Only MultiStepLR and CosineAnnealingLR "
            "are supported.".format(args.lr_scheduler)
        )

    if args.resume:
        checkpoint = flow.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1

    if args.test_only:
        evaluate(model, data_loader_test, device=device)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": args,
                "epoch": epoch,
            }
            utils.save_on_master(
                checkpoint, os.path.join(args.output_dir, "model_{}.pth".format(epoch))
            )
            utils.save_on_master(
                checkpoint, os.path.join(args.output_dir, "checkpoint.pth")
            )

        # evaluate after every epoch
        evaluate(model, data_loader_test, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
