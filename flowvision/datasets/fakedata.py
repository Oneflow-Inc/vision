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
from typing import Any, Callable, Optional, Tuple

import oneflow as flow

from .. import transforms
from .vision import VisionDataset


class FakeData(VisionDataset):
    """A fake dataset that returns randomly generated images and returns them as PIL images
    Args:
        size (int, optional): Size of the dataset. Default: 1000 images
        image_size(tuple, optional): Size if the returned images. Default: (3, 224, 224)
        num_classes(int, optional): Number of classes in the dataset. Default: 10
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        random_offset (int): Offsets the index-based random seed used to
            generate each image. Default: 0
    """

    def __init__(
        self,
        size: int = 1000,
        image_size: Tuple[int, int, int] = (3, 224, 224),
        num_classes: int = 10,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        random_offset: int = 0,
    ) -> None:
        super(FakeData, self).__init__(
            None, transform=transform, target_transform=target_transform  # type: ignore[arg-type]
        )
        self.size = size
        self.num_classes = num_classes
        self.image_size = image_size
        self.random_offset = random_offset

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # create random image that is consistent with the index id
        if index >= len(self):
            raise IndexError("{} index out of range".format(self.__class__.__name__))
        # rng_state = flow.get_rng_state()
        rng_state = flow.default_generator().get_state()
        flow.manual_seed(index + self.random_offset)
        img = flow.randn(*self.image_size)
        target = flow.randint(0, self.num_classes, size=(1,), dtype=flow.long)[0]
        # flow.set_rng_state(rng_state)
        flow.default_generator().set_state(rng_state)

        # convert to PIL Image
        img = transforms.ToPILImage()(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target.item()

    def __len__(self) -> int:
        return self.size
