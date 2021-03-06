import sys
import torch.utils.data as data
import torchvision.transforms as transforms
import copy
import numpy as np
from .samples import SysuSamples, RegDBSamples
from .sampler import UniformSampler, Seeds, CrossUniformSampler
from .loader import ReIDDataSet, IterLoader, CrossDataset
from .random_erasing import RandomErasing


class Loaders:
    def __init__(self, config):

        # transforms

        self.transform_test = transforms.Compose(
            [
                transforms.Resize(config.image_size, interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                    # mean=[0.5] * 3, std=[0.5] * 3
                ),  # we normalize it in reid model with imagenet mean and std
            ]
        )

        self.transform_reid = transforms.Compose(
            [
                transforms.Resize(config.image_size, interpolation=3),
                transforms.Pad(10),
                transforms.RandomCrop(config.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                    # mean=[0.5] * 3, std=[0.5] * 3
                )  # we normalize it in reid model with imagenet mean and std
                # RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
            ]
        )

        #  dataset configuration
        self.dataset_path = config.dataset_path
        self.dataset = config.dataset

        # sample configuration
        self.p = config.p
        self.k = config.k

        # init loaders
        self._init_train_loaders()

    def _init_train_loaders(self):
        if self.dataset == "sysu":
            all_samples = SysuSamples(self.dataset_path, True)
        else:
            all_samples = RegDBSamples(self.dataset_path, True)

        # init datasets
        rgb_ir_train_dataset = CrossDataset(
            all_samples.rgb_samples_train,
            all_samples.ir_samples_train,
            self.transform_reid,
        )
        rgb_test_dataset = ReIDDataSet(
            all_samples.rgb_samples_test, self.transform_test
        )
        in_door_rgb_test_dataset = ReIDDataSet(
            all_samples.in_door_rgb_samples_test, self.transform_test
        )
        ir_test_dataset = ReIDDataSet(all_samples.ir_samples_test, self.transform_test)
        rgb_all_dataset = ReIDDataSet(all_samples.rgb_samples_all, self.transform_test)
        ir_all_dataset = ReIDDataSet(all_samples.ir_samples_all, self.transform_test)

        # init loaders
        seeds = Seeds(np.random.randint(0, 1e8, 9999))

        # reid train dataset
        self.rgb_ir_train_loader = data.DataLoader(
            copy.deepcopy(rgb_ir_train_dataset),
            self.p * self.k,
            shuffle=False,
            sampler=CrossUniformSampler(rgb_ir_train_dataset, self.k, copy.copy(seeds)),
            num_workers=16,
            drop_last=True,
        )

        # init iters
        # self.reid_rgb_ir_train_iter = IterLoader(self.rgb_ir_train_loader)

        # test dataset
        self.rgb_test_loader = data.DataLoader(
            rgb_test_dataset, 32, shuffle=False, num_workers=8, drop_last=False
        )
        self.in_door_rgb_test_loader = data.DataLoader(
            in_door_rgb_test_dataset, 32, shuffle=False, num_workers=8, drop_last=False
        )
        self.ir_test_loader = data.DataLoader(
            ir_test_dataset, 32, shuffle=False, num_workers=8, drop_last=False
        )

        self.rgb_all_loader = data.DataLoader(
            rgb_all_dataset, 128, shuffle=False, num_workers=8, drop_last=False
        )
        self.ir_all_loader = data.DataLoader(
            ir_all_dataset, 128, shuffle=False, num_workers=8, drop_last=False
        )

