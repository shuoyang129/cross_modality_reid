import torch
from bisect import bisect_right


class CustomMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self, optimizer, warmup_epochs, milestones, gamma=0.1, last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )
        self.warmup_epochs = warmup_epochs
        self.milestones = milestones
        self.gamma = gamma
        super(CustomMultiStepLR, self).__init__(optimizer, last_epoch)

    # if epoch < 10:
    #         lr = args.lr * (epoch + 1) / 10
    #     elif epoch >= 10 and epoch < 20:
    #         lr = args.lr
    #     elif epoch >= 20 and epoch < 50:
    #         lr = args.lr * 0.1
    #     elif epoch >= 50:
    #         lr = args.lr * 0.01
    #  the epoch legth of original code is 10 times of now

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_epochs:
            warmup_factor = (self.last_epoch // 10 + 1) / (self.warmup_epochs // 10)
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]
