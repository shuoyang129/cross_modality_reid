"""
@author:    Guan'an Wang
@contact:   guan.wang0706@gmail.com
"""

import numpy as np
import os
import time
import torch
import torch.nn as nn
from prettytable import PrettyTable

from evaluations import eval_sysu, eval_regdb, accuracy
from utils import (
    MultiItemAverageMeter,
    CatMeter,
    AverageMeter,
    Logging,
    time_now,
    os_walk,
)


class Engine(object):
    def __init__(
        self,
        results_dir,
        dataloaders,
        model,
        criterion,
        optimizer,
        use_gpu,
        test_dataset,
        test_mode,
        data_parallel=False,
        sync_bn=False,
    ):

        # base settings
        self.results_dir = os.path.join(results_dir, "")
        self.dataloaders = dataloaders
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")
        self.test_dataset = test_dataset
        self.test_mode = test_mode
        assert test_mode in ["all", "in_door"], "test_mode should be 'all' or 'in_door'"

        self.loss_meter = MultiItemAverageMeter()
        os.makedirs(self.results_dir, exist_ok=True)
        self.logging = Logging(os.path.join(self.results_dir, "logging.txt"))

        self.model = self.model.to(self.device)
        if data_parallel:
            if not sync_bn:
                self.model = nn.DataParallel(self.model)
            if sync_bn:
                # torch.distributed.init_process_group(backend='nccl', world_size=4, init_method='...')
                self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
                # self.model = nn.parallel.DistributedDataParallel(self.model)

    def save_model(self, save_epoch):
        """
        save model parameters (only state_dict) in self.results_dir/model_{epoch}.pth
        save model (architecture and state_dict) in self.results_dir/final_model.pth.tar, may be used as a teacher
        """
        model_path = os.path.join(self.results_dir, "model_{}.pth".format(save_epoch))
        torch.save(self.model.state_dict(), model_path)
        root, _, files = os_walk(self.results_dir)
        pth_files = [
            file for file in files if ".pth" in file and file != "final_model.pth.tar"
        ]
        if len(pth_files) > 1:
            pth_epochs = sorted(
                [
                    int(pth_file.replace(".pth", "").split("_")[1])
                    for pth_file in pth_files
                ],
                reverse=False,
            )
            model_path = os.path.join(root, "model_{}.pth".format(pth_epochs[0]))
            os.remove(model_path)
        torch.save(self.model, os.path.join(self.results_dir, "final_model.pth.tar"))

    def resume_model(self, model_path, strict=True):
        """
        resume from model_path
        """
        self.model.load_state_dict(torch.load(model_path), strict=strict)

    def resume_latest_model(self):
        """
        resume from the latest model in path self.results_dir
        """
        root, _, files = os_walk(self.results_dir)
        pth_files = [
            file for file in files if ".pth" in file and file != "final_model.pth.tar"
        ]
        if len(pth_files) != 0:
            pth_epochs = [
                int(pth_file.replace(".pth", "").split("_")[1])
                for pth_file in pth_files
            ]
            max_epoch = max(pth_epochs)
            model_path = os.path.join(root, "model_{}.pth".format(max_epoch))
            self.model.load_state_dict(torch.load(model_path), strict=True)
            self.logging(time_now(), "restore from {}".format(model_path))
            return max_epoch
        else:
            return None

    def set_train(self):
        """
        set model as training mode
        """
        self.model = self.model.train()

    def set_eval(self):
        """
        set mode as evaluation model
        """
        self.model = self.model.eval()

    def train(self, auto_resume=True, eval_freq=0):
        """
        Args:
            auto_resume(boolean): automatically resume latest model from self.result_dir/model_{latest_epoch}.pth if True.
            eval_freq(int): if type is int, evaluate every eval_freq. default is 0.
        """

        # automatically resume from the latest model
        start_epoch = 0
        if auto_resume:
            start_epoch = self.resume_latest_model()
            start_epoch = 0 if start_epoch is None else start_epoch
        # train loop
        for curr_epoch in range(start_epoch, self.optimizer.max_epochs):
            # save model
            self.save_model(curr_epoch)
            # evaluate final model
            if eval_freq > 0 and curr_epoch % eval_freq == 0 and curr_epoch > 0:
                self.eval(self.test_dataset)
            # train
            results = self.train_an_epoch(curr_epoch)
            # logging
            self.logging(EPOCH=curr_epoch, TIME=time_now(), RESULTS=results)
        # save final model
        self.save_model(self.optimizer.max_epochs)
        # evaluate final model
        self.eval(self.test_dataset)

    def train_an_epoch(self, epoch):
        self.set_train()
        self.loss_meter.reset()
        for batch_idx, (rgb_data, ir_data) in enumerate(
            self.dataloaders.rgb_ir_train_loader
        ):
            input1, label1, _, _ = rgb_data
            input2, label2, _, _ = ir_data
            labels = torch.cat((label1, label2), 0)

            input1 = input1.to(self.device)
            input2 = input2.to(self.device)
            labels = labels.to(self.device)

            feat, bnfeat, logit = self.model(input1, input2)

            acc = accuracy(logit, labels, [1])[0]
            loss, loss_dict = self.criterion.compute(
                feats=feat, head_feats=bnfeat, logits=logit, pids=labels
            )

            loss_dict["Accuracy"] = acc
            # optimize
            self.optimizer.optimizer.zero_grad()
            loss.backward()
            self.optimizer.optimizer.step()
            # update learning rate
            self.optimizer.lr_scheduler.step(epoch)
            # record
            self.loss_meter.update(loss_dict)

            # if batch_idx % 20 == 0:
            #     self.logging(
            #         EPOCH=epoch,
            #         BATCH="{}/{}".format(
            #             batch_idx, len(self.dataloaders.rgb_ir_train_loader)
            #         ),
            #         TIME=time_now(),
            #         RESULTS=self.loss_meter.get_str(),
            #     )

        # learning rate
        self.loss_meter.update({"LR": self.optimizer.optimizer.param_groups[0]["lr"]})

        return self.loss_meter.get_str()

    def eval(self, dataset, trial=1):

        self.set_eval()

        table = PrettyTable(
            ["dataset", "feature", "map", "rank-1", "rank-5", "rank-10", "mINP"]
        )

        # evaluation
        if dataset == "regdb":
            # gallery：IR 2 , query: RGB 1
            query_feat_pool, query_feat_fc, query_label, _ = self.extract_features(
                self.dataloaders.rgb_test_loader, 1
            )
            gall_feat_pool, gall_feat_fc, gallery_label, _ = self.extract_features(
                self.dataloaders.ir_test_loader, 2
            )
            distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
            distmat_fc = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))

            cmc, mAP, mINP = eval_regdb(-distmat_pool, query_label, gallery_label)
            cmc_fc, mAP_fc, mINP_fc = eval_regdb(
                -distmat_fc, query_label, gallery_label
            )
        elif dataset == "sysu":
            # gallery：RGB 1  , query: IR 2
            (
                query_feat_pool,
                query_feat_fc,
                query_label,
                query_cam,
            ) = self.extract_features(self.dataloaders.ir_test_loader, 2)
            if self.test_mode == "all":
                (
                    gall_feat_pool,
                    gall_feat_fc,
                    gallery_label,
                    gall_cam,
                ) = self.extract_features(self.dataloaders.rgb_test_loader, 1)
            else:
                (
                    gall_feat_pool,
                    gall_feat_fc,
                    gallery_label,
                    gall_cam,
                ) = self.extract_features(self.dataloaders.in_door_rgb_test_loader, 1)

            distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
            distmat_fc = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))

            cmc, mAP, mINP = eval_sysu(
                -distmat_pool, query_label, gallery_label, query_cam, gall_cam
            )
            cmc_fc, mAP_fc, mINP_fc = eval_sysu(
                -distmat_fc, query_label, gallery_label, query_cam, gall_cam
            )
            # logging
            table.add_row(
                [
                    dataset,
                    "pooling",
                    str(mAP),
                    str(cmc[0]),
                    str(cmc[4]),
                    str(cmc[9]),
                    str(mINP),
                ]
            )
            table.add_row(
                [
                    dataset,
                    "bn",
                    str(mAP_fc),
                    str(cmc_fc[0]),
                    str(cmc_fc[4]),
                    str(cmc_fc[9]),
                    str(mINP_fc),
                ]
            )
            self.logging(mAP_fc, cmc_fc[:150])
        self.logging(table)

    def extract_features(self, loader, modality, time_meter=None):
        # modality 1: visible 2: thermal
        self.set_eval()
        # compute features
        features_meter = None
        bn_features_meter = None
        pids_meter = CatMeter()
        cids_meter = CatMeter()
        with torch.no_grad():
            for batch in loader:
                imgs, pids, cids, _ = batch
                imgs, pids, cids = (
                    imgs.to(self.device),
                    pids.to(self.device),
                    cids.to(self.device),
                )
                if time_meter is not None:
                    torch.cuda.synchronize()
                    ts = time.time()
                feat_pool, feat_fc = self.model(imgs, imgs, modality)
                if time_meter is not None:
                    torch.cuda.synchronize()
                    time_meter.update(time.time() - ts)
                if isinstance(feat_pool, torch.Tensor):
                    if features_meter is None:
                        features_meter = CatMeter()
                        bn_features_meter = CatMeter()
                    features_meter.update(feat_pool.data)
                    bn_features_meter.update(feat_fc.data)
                elif isinstance(feat_pool, list):
                    if features_meter is None:
                        features_meter = [CatMeter() for _ in range(len(feat_pool))]
                        bn_features_meter = [CatMeter() for _ in range(len(feat_pool))]
                    for idx in range(len(feat_pool)):
                        features_meter[idx].update(feat_pool[idx].data)
                        bn_features_meter[idx].update(feat_fc[idx].data)
                else:
                    assert 0
                pids_meter.update(pids.data)
                cids_meter.update(cids.data)

        if isinstance(features_meter, list):
            feats = [val.get_val_numpy() for val in features_meter]
            bn_feats = [val.get_val_numpy() for val in bn_features_meter]
        else:
            feats = features_meter.get_val_numpy()
            bn_feats = bn_features_meter.get_val_numpy()
        pids = pids_meter.get_val_numpy()
        cids = cids_meter.get_val_numpy()

        return feats, bn_feats, pids, cids

