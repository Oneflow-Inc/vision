from .mnist import MNIST, FashionMNIST
from .caltech import Caltech101, Caltech256
from .celeba import CelebA
from .cifar import CIFAR10, CIFAR100
from .cityscapes import Cityscapes
from .clevr import CLEVRClassification
from .country211 import Country211
from .coco import CocoCaptions, CocoDetection
from .dtd import DTD
from .eurosat import EuroSAT
from .gtsrb import GTSRB
from .imagenet import ImageNet
from .voc import VOCDetection, VOCSegmentation
from .folder import DatasetFolder, ImageFolder
from .fakedata import FakeData
from .fgvc_aircraft import FGVCAircraft
from .flowers102 import Flowers102
from .flickr import Flickr8k, Flickr30k
from .food101 import Food101
from .inaturalist import INaturalist
from .kitti import Kitti
from .lfw import LFWPairs, LFWPeople
from .lsun import LSUN, LSUNClass
from .omniglot import Omniglot
from .oxford_iiit_pet import OxfordIIITPet
from .pcam import PCAM
from .phototour import PhotoTour
from .places365 import Places365
from .rendered_sst2 import RenderedSST2
from .sbd import SBDataset
from .sbu import SBU
from .semeion import SEMEION
from .stanford_cars import StanfordCars
from .stl10 import STL10
from .svhn import SVHN
from .sun397 import SUN397
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
    "CLEVRClassification",
    "Country211",
    "CocoCaptions",
    "CocoDetection",
    "ImageNet",
    "VOCDetection",
    "VOCSegmentation",
    "DatasetFolder",
    "DTD",
    "EuroSAT",
    "GTSRB",
    "ImageFolder",
    "FakeData",
    "FGVCAircraft",
    "Flowers102",
    "Flickr8k",
    "Flickr30k",
    "Food101",
    "INaturalist",
    "Kitti",
    "LFWPairs",
    "LFWPeople",
    "LSUN",
    "LSUNClass",
    "Omniglot",
    "OxfordIIITPet",
    "PCAM",
    "PhotoTour",
    "Places365",
    "RenderedSST2",
    "SBDataset",
    "SBU",
    "SUN397",
    "SEMEION",
    "StanfordCars",
    "STL10",
    "SVHN",
    "USPS",
    "VisionDataset",
    "WIDERFace",
]
