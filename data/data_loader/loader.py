from PIL import Image
import copy


class ReIDDataSet:
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):

        this_sample = copy.deepcopy(self.samples[index])
        this_sample[0] = self._loader(this_sample[0])
        if self.transform is not None:
            this_sample[0] = self.transform(this_sample[0])

        return this_sample

    def __len__(self):
        return len(self.samples)

    def _loader(self, img_path):
        return Image.open(img_path).convert("RGB")


class IterLoader:
    def __init__(self, loader):
        self.loader = loader
        self.iter = iter(self.loader)

    def next_one(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)


class CrossDataset:
    def __init__(self, rgb_samples, ir_samples, transform):
        self.rgb_samples = rgb_samples
        self.ir_samples = ir_samples
        self.transform = transform

    def __getitem__(self, index):
        rgb_index, ir_index = index
        this_sample1 = copy.deepcopy(self.rgb_samples[rgb_index])
        this_sample2 = copy.deepcopy(self.ir_samples[ir_index])
        this_sample1[0] = self._loader(this_sample1[0])
        this_sample2[0] = self._loader(this_sample2[0])
        if self.transform is not None:
            this_sample1[0] = self.transform(this_sample1[0])
            this_sample2[0] = self.transform(this_sample2[0])

        return this_sample1, this_sample2

    def __len__(self):
        return len(self.rgb_samples)

    def _loader(self, img_path):
        return Image.open(img_path).convert("RGB")

