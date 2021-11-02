"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from .mnist import MNIST, FashionMNIST
from .caltech import Caltech101, Caltech256
from .celeba import CelebA
from .cifar import CIFAR10, CIFAR100
from .cityscapes import Cityscapes
from .coco import CocoCaptions, CocoDetection
from .imagenet import ImageNet
from .voc import VOCDetection, VOCSegmentation
from .folder import DatasetFolder, ImageFolder
from .fakedata import FakeData
from .flickr import Flickr8k, Flickr30k
from .inaturalist import INaturalist
from .kitti import Kitti
from .lfw import LFWPairs, LFWPeople
from .lsun import LSUN, LSUNClass
from .omniglot import Omniglot
from .phototour import PhotoTour
from .places365 import Places365
from .sbd import SBDataset
from .sbu import SBU
from .semeion import SEMEION
from .stl10 import STL10
from .svhn import SVHN
from .usps import USPS
from .vision import VisionDataset
from .widerface import WIDERFace

__all__ = [
    "MNIST",
    "FashionMNIST",
    "CIFAR10",
    "CIFAR100",
    "Caltech101",
    "Caltech256",
    "CelebA",
    "Cityscapes",
    "CocoCaptions",
    "CocoDetection",
    "ImageNet",
    "VOCDetection",
    "VOCSegmentation",
    "DatasetFolder",
    "ImageFolder",
    "FakeData",
    "Flickr8k",
    "Flickr30k",
    "INaturalist",
    "Kitti",
    "LFWPairs",
    "LFWPeople",
    "LSUN",
    "LSUNClass",
    "Omniglot",
    "PhotoTour",
    "Places365",
    "SBDataset",
    "SBU",
    "SEMEION",
    "STL10",
    "SVHN",
    "USPS",
    "VisionDataset",
    "WIDERFace",
]
