import numpy as np
import os
import time
import scipy.io as sio
import torch
import torch.nn as nn
from prettytable import PrettyTable

from evaluations import eval_sysu, eval_regdb, accuracy, evaluate_sysymm01
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
        seed,
        data_parallel=False,
        sync_bn=False,
    ):

        # base settings
        self.results_dir = results_dir
        self.dataloaders = dataloaders
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")
        self.test_dataset = test_dataset
        self.test_mode = test_mode
        self.seed = seed
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
        best_epoch = start_epoch
        best_rank1 = -100
        best_rank1_map = -1
        for curr_epoch in range(start_epoch, self.optimizer.max_epochs):
            # save model
            self.save_model(curr_epoch)
            # evaluate final model
            if eval_freq > 0 and curr_epoch % eval_freq == 0 and curr_epoch > 0:
                cmc, mAP = self.eval(self.test_dataset)
                # self.eval2(self.test_dataset)
                if cmc[0] > best_rank1:
                    best_epoch = curr_epoch
                    best_rank1 = cmc[0]
                    best_rank1_map = mAP
                print(
                    "best rank1: ",
                    best_rank1,
                    "mAP:",
                    best_rank1_map,
                    "epoch: ",
                    best_epoch,
                )
            # train
            results = self.train_an_epoch(curr_epoch)
            # logging
            self.logging(EPOCH=curr_epoch, TIME=time_now(), RESULTS=results)
        # save final model
        self.save_model(self.optimizer.max_epochs)
        # evaluate final model
        self.eval(self.test_dataset)
        # self.eval2(self.test_dataset)

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
                print("test mode:  all")
                (
                    gall_feat_pool,
                    gall_feat_fc,
                    gallery_label,
                    gall_cam,
                ) = self.extract_features(self.dataloaders.rgb_test_loader, 1)
            else:
                print("test mode:  in_door")
                (
                    gall_feat_pool,
                    gall_feat_fc,
                    gallery_label,
                    gall_cam,
                ) = self.extract_features(self.dataloaders.in_door_rgb_test_loader, 1)

            # the baseline code(original code) is random sample one image of every camera of the person, so here should be a filter to choose a sample per camera per person
            np.random.seed(self.seed)
            choose_indexs = []
            unique_label = np.unique(gallery_label)
            for label in unique_label:
                index = np.where(gallery_label == label)[0]
                person_all_cam = gall_cam[index]
                cam_unique = np.unique(person_all_cam)
                for cam in cam_unique:
                    all_i = np.where(person_all_cam == cam)[0]
                    choose_i = np.random.choice(all_i)
                    choose_indexs.append(index[choose_i])
            choose_indexs = np.array(choose_indexs)
            gall_feat_pool = gall_feat_pool[choose_indexs]
            gall_feat_fc = gall_feat_fc[choose_indexs]
            gallery_label = gallery_label[choose_indexs]
            gall_cam = gall_cam[choose_indexs]

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
                    "%.2f" % (mAP * 100),
                    "%.2f" % (cmc[0] * 100),
                    "%.2f" % (cmc[4] * 100),
                    "%.2f" % (cmc[9] * 100),
                    "%.2f" % (mINP * 100),
                ]
            )
            table.add_row(
                [
                    dataset,
                    "bn",
                    "%.2f" % (mAP_fc * 100),
                    "%.2f" % (cmc_fc[0] * 100),
                    "%.2f" % (cmc_fc[4] * 100),
                    "%.2f" % (cmc_fc[9] * 100),
                    "%.2f" % (mINP_fc * 100),
                ]
            )
            self.logging(mAP, cmc[:150])
            self.logging(mAP_fc, cmc_fc[:150])
        self.logging(table)
        return cmc_fc, mAP_fc

    def extract_features(self, data_loader, modality, time_meter=None):
        # modality 1: visible 2: thermal
        self.set_eval()
        # compute features
        features_meter = None
        bn_features_meter = None
        pids_meter = CatMeter()
        cids_meter = CatMeter()
        with torch.no_grad():
            for batch in data_loader:
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

    def eval2(self, dataset, brief=False):
        self.set_eval()
        if dataset != "sysu":
            assert 0, " this test methods only used for sysu dataset"
        self.compute_and_save_features(self.dataloaders)
        results = {}
        for mode in ["all", "indoor"]:
            for number_shot in ["single", "multi"]:
                cmc, mAP = evaluate_sysymm01(
                    os.path.join(self.results_dir, "test_features"), mode, number_shot
                )
                results["{},{}".format(mode, number_shot)] = [cmc, mAP]
                if brief:
                    break
            if brief:
                break
        table = PrettyTable(
            ["mode,shot", "feature", "map", "rank-1", "rank-5", "rank-10"]
        )
        keys = results.keys()
        for key in keys:
            cmc, mAP = results[key]
            table.add_row(
                [
                    key,
                    "bn",
                    "%.2f" % (mAP * 100),
                    "%.2f" % (cmc[0] * 100),
                    "%.2f" % (cmc[4] * 100),
                    "%.2f" % (cmc[9] * 100),
                ]
            )
        self.logging(table)
        # for key in results.keys():
        #     self.logging(
        #         "Time: {}\n Setting: {}\n {}".format(time_now(), key, results[key])
        #     )

    def compute_and_save_features(self, loaders):
        print("Time:{}.  Start to compute features".format(time_now()))
        # compute features
        features_meter, pids_meter, cids_meter = CatMeter(), CatMeter(), CatMeter()
        self.set_eval()
        with torch.no_grad():
            for i, data in enumerate(loaders.rgb_all_loader):
                # load data
                images, pids, cids, _ = data
                # forward
                images = images.to(self.device)
                feat_pool, feat_fc = self.model(images, images, 1)
                # meter
                features_meter.update(feat_fc.data)
                pids_meter.update(pids.data)
                cids_meter.update(cids.data)

            for i, data in enumerate(loaders.ir_all_loader):
                # load data
                images, pids, cids, _ = data
                # forward
                images = images.to(self.device)
                feat_pool, feat_fc = self.model(images, images, 2)
                # meter
                features_meter.update(feat_fc.data)
                pids_meter.update(pids.data)
                cids_meter.update(cids.data)

        features = features_meter.get_val_numpy()
        pids = pids_meter.get_val_numpy()
        cids = cids_meter.get_val_numpy()

        print("Time: {}.  Note: Start to save features as .mat file".format(time_now()))
        # save features as .mat file
        results = {1: XX(), 2: XX(), 3: XX(), 4: XX(), 5: XX(), 6: XX()}
        for i in range(features.shape[0]):
            feature = features[i, :]
            feature = np.resize(feature, [1, feature.shape[0]])
            cid, pid = cids[i], pids[i]
            results[cid].update(pid, feature)

        pid_num_of_cids = [333, 333, 533, 533, 533, 333]
        cids = [1, 2, 3, 4, 5, 6]
        for cid in cids:
            a_result = results[cid]
            xx = []
            for pid in range(1, 1 + pid_num_of_cids[cid - 1]):
                xx.append([a_result.get_val(pid).astype(np.double)])
            xx = np.array(xx)
            os.makedirs(os.path.join(self.results_dir, "test_features"), exist_ok=True)
            sio.savemat(
                os.path.join(
                    self.results_dir, "test_features", "feature_cam{}.mat".format(cid)
                ),
                {"feature": xx},
            )
        print("Time: {}. end to save features as .mat file".format(time_now()))


class XX:
    def __init__(self):
        self.val = {}

    def update(self, key, value):
        if key not in list(self.val.keys()):
            self.val[key] = value
        else:
            self.val[key] = np.concatenate([self.val[key], value], axis=0)

    def get_val(self, key):
        if key in list(self.val.keys()):
            return self.val[key]
        else:
            return np.array([[]])

