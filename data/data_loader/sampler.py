import random
import torch.utils.data as data
import numpy as np


class UniformSampler(data.sampler.Sampler):
    def __init__(self, dataset, k, random_seeds):

        self.dataset = dataset
        self.k = k
        self.random_seeds = random_seeds

        self.samples = self.dataset.samples
        self._process()

        self.sample_list = self._generate_list()

    def __iter__(self):
        self.sample_list = self._generate_list()
        return iter(self.sample_list)

    def __len__(self):
        return len(self.sample_list)

    def _process(self):
        pids, cids = [], []
        for sample in self.samples:
            _, pid, cid, _ = sample
            pids.append(pid)
            cids.append(cid)

        self.pids = np.array(pids)
        self.cids = np.array(cids)

    def _generate_list(self):

        index_list = []
        pids = list(set(self.pids))
        pids.sort()

        seed = self.random_seeds.next_one()
        random.seed(seed)
        random.shuffle(pids)

        for pid in pids:
            # find all indexes of the person of pid
            index_of_pid = np.where(self.pids == pid)[0]
            # randomly sample k images from the pid
            if len(index_of_pid) >= self.k:
                index_list.extend(
                    np.random.choice(index_of_pid, self.k, replace=False).tolist()
                )
            else:
                index_list.extend(
                    np.random.choice(index_of_pid, self.k, replace=True).tolist()
                )

        return index_list


class Seeds:
    def __init__(self, seeds):
        self.index = -1
        self.seeds = seeds

    def next_one(self):
        self.index += 1
        if self.index > len(self.seeds) - 1:
            self.index = 0
        return self.seeds[self.index]


class CrossUniformSampler(data.sampler.Sampler):
    def __init__(self, dataset, k, random_seeds):

        self.dataset = dataset
        self.k = k
        self.random_seeds = random_seeds

        self.rgb_samples = self.dataset.rgb_samples
        self.ir_samples = self.dataset.ir_samples
        self._process()

        self.sample_list = self._generate_list()

    def __iter__(self):
        self.sample_list = self._generate_list()
        return iter(self.sample_list)

    def __len__(self):
        return len(self.sample_list)

    def _process(self):
        rgb_pids, rgb_cids, rgb_mids = [], [], []  # person id, camera id, modality id
        for sample in self.rgb_samples:
            _, pid, cid, mid = sample
            rgb_pids.append(pid)
            rgb_cids.append(cid)
            rgb_mids.append(mid)

        self.rgb_pids = np.array(rgb_pids)
        self.rgb_cids = np.array(rgb_cids)
        self.rgb_mids = np.array(rgb_mids)

        ir_pids, ir_cids, ir_mids = [], [], []  # person id, camera id, modality id
        for sample in self.ir_samples:
            _, pid, cid, mid = sample
            ir_pids.append(pid)
            ir_cids.append(cid)
            ir_mids.append(mid)

        self.ir_pids = np.array(ir_pids)
        self.ir_cids = np.array(ir_cids)
        self.ir_mids = np.array(ir_mids)

    def _generate_list(self):

        rgb_index_list = []
        ir_index_list = []
        index_list = []
        pids = list(set(self.pids))
        pids.sort()

        seed = self.random_seeds.next_one()
        random.seed(seed)
        random.shuffle(pids)

        for pid in pids:
            # find all indexes of the person of pid
            index_of_pid_of_all = np.where(self.pids == pid)[0]

            index_of_index = np.where(self.mids[index_of_pid_of_all] == 0)[
                0
            ]  # rgb index of index_of_pid
            index_of_pid = index_of_pid_of_all[index_of_index]  # index of RGB
            # randomly sample k images from the pid
            if len(index_of_pid) >= self.k:
                rgb_index_list.extend(
                    np.random.choice(index_of_pid, self.k, replace=False).tolist()
                )
            else:
                rgb_index_list.extend(
                    np.random.choice(index_of_pid, self.k, replace=True).tolist()
                )

            index_of_index = np.where(self.mids[index_of_pid_of_all] == 1)[
                0
            ]  # ir index of index_of_pid
            index_of_pid = index_of_pid_of_all[index_of_index]  # index of IR
            # randomly sample k images from the pid
            if len(index_of_pid) >= self.k:
                ir_index_list.extend(
                    np.random.choice(index_of_pid, self.k, replace=False).tolist()
                )
            else:
                ir_index_list.extend(
                    np.random.choice(index_of_pid, self.k, replace=True).tolist()
                )

            assert len(rgb_index_list) == len(ir_index_list), "error of sampling"
            for rgb, ir in zip(ir_index_list, ir_index_list):
                index_list.extend((rgb, ir))
        return index_list
