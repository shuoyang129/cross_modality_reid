import torch
import time
import numpy as np
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from datasets import SYSUData, RegDBData, TestData
from data_manager import (
    process_test_regdb,
    process_query_sysu,
    process_gallery_sysu,
)
from sampler import IdentitySampler
from utils import GenIdx, GenCamIdx, ExtractCam


class CrossModalLoaders:
    def __init__(self, config, transform_train, transform_test):
        self.config = config
        super(CrossModalLoaders, self).__init__()

        self.transform_train = transform_train
        self.transform_test = transform_test

        self._load_data()

    def _load_data(self):
        start = time.time()
        assert self.config.dataset in ["sysu", "regdb"]
        query_cam = None
        gallery_cam = None
        if self.config.dataset == "sysu":
            # training set
            trainset = SYSUData(self.config.data_path, transform=self.transform_train)
            # generate the idx of each person identity
            color_pos, thermal_pos = GenIdx(
                trainset.train_color_label, trainset.train_thermal_label
            )

            # testing set
            query_img, query_label, query_cam = process_query_sysu(
                self.config.data_path, mode=self.config.mode
            )
            gall_img, gall_label, gallery_cam = process_gallery_sysu(
                self.config.data_path, mode=self.config.mode, trial=0
            )

        else:
            # regdb dataset
            # training set
            trainset = RegDBData(
                self.config.data_path, self.config.trial, transform=self.transform_train
            )
            # generate the idx of each person identity
            color_pos, thermal_pos = GenIdx(
                trainset.train_color_label, trainset.train_thermal_label
            )

            # testing set
            query_img, query_label = process_test_regdb(
                self.config.data_path, trial=self.config.trial, modal="visible"
            )
            gall_img, gall_label = process_test_regdb(
                self.config.data_path, trial=self.config.trial, modal="thermal"
            )

        gallset = TestData(
            gall_img,
            gall_label,
            transform=self.transform_test,
            img_size=(self.config.img_w, self.config.img_h),
        )
        queryset = TestData(
            query_img,
            query_label,
            transform=self.transform_test,
            img_size=(self.config.img_w, self.config.img_h),
        )

        # testing data loader
        gallery_loader = data.DataLoader(
            gallset,
            batch_size=self.config.test_batch,
            shuffle=False,
            num_workers=self.config.workers,
        )
        query_loader = data.DataLoader(
            queryset,
            batch_size=self.config.test_batch,
            shuffle=False,
            num_workers=self.config.workers,
        )
        n_class = len(np.unique(trainset.train_color_label))
        nquery = len(query_label)
        ngall = len(gall_label)

        print("Dataset {} statistics:".format(self.config.dataset))
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print(
            "  visible  | {:5d} | {:8d}".format(
                n_class, len(trainset.train_color_label)
            )
        )
        print(
            "  thermal  | {:5d} | {:8d}".format(
                n_class, len(trainset.train_thermal_label)
            )
        )
        print("  ------------------------------")
        print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
        print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
        print("  ------------------------------")
        print("Data Loading Time:\t {:.3f}".format(time.time() - start))

        sampler = IdentitySampler(
            trainset.train_color_label,
            trainset.train_thermal_label,
            color_pos,
            thermal_pos,
            self.config.num_pos,
            self.config.batch_size,
        )

        trainset.cIndex = sampler.index1  # color index
        trainset.tIndex = sampler.index2  # thermal index

        loader_batch = self.config.batch_size * self.config.num_pos

        trainloader = data.DataLoader(
            trainset,
            batch_size=loader_batch,
            sampler=sampler,
            num_workers=self.config.workers,
            drop_last=True,
        )

        self.trainloader = trainloader
        self.query_loader = query_loader
        self.gallery_loader = gallery_loader
        self.query_cam = query_cam
        self.gallery_cam = gallery_cam

