from data.data_loader import loader
import os
import sys
import torch
import torch.nn as nn
import ast
import argparse

from models import embed_net
from utils import *
from losses import CrossEntropyLabelSmooth, TripletLoss, Criterion
from optim import (
    Optimizer,
    WarmupMultiStepLR,
    DelayedCosineAnnealingLR,
    WarmupCosineAnnealingLR,
)
from evaluations import eval_sysu, eval_regdb, accuracy
from data import Loaders
from engine import Engine

# Settings
parser = argparse.ArgumentParser(description="PyTorch Cross-Modality Training")
parser.add_argument(
    "--results_dir",
    type=str,
    default="/home/Monday/results/",
    help="path to save outputs",
)
parser.add_argument("--img_w", default=144, type=int, metavar="imgw", help="img width")
parser.add_argument("--img_h", default=288, type=int, metavar="imgh", help="img height")
parser.add_argument(
    "--p", default=8, type=int, metavar="P", help="persons of each batch"
)
parser.add_argument(
    "--k", default=4, type=int, metavar="K", help="images of each person in batch",
)
parser.add_argument(
    "--dataset_root",
    default="/home/Monday/datasets",
    help="dataset root where the sysu and regdb put",
)
parser.add_argument(
    "--dataset", default="sysu", help="train dataset name: regdb or sysu"
)
parser.add_argument(
    "--test_dataset", type=str, default="sysu", help="test dataset name: regdb or sysu"
)
parser.add_argument("--test_mode", default="all", type=str, help="all or indoor")
parser.add_argument(
    "--arch", default="resnet50", type=str, help="network baseline:resnet18 or resnet50"
)
parser.add_argument(
    "--pooling_type",
    default=1,
    type=int,
    help="pooling_type:0-->avgpooling, 1-->gm_pooling, 2-->similarity,3-->avgpooling+similarity, 4--> gm_pooling+similarity",
)
parser.add_argument("--optim", default="sgd", type=str, help="optimizer")
parser.add_argument(
    "--lr", default=0.00035, type=float, help="learning rate, 0.00035 for adam"
)
parser.add_argument(
    "--margin", default=0.3, type=float, metavar="margin", help="triplet loss margin"
)
parser.add_argument(
    "--resume", "-r", default="", type=str, help="resume from checkpoint"
)
parser.add_argument(
    "--gpu", default="0", type=str, help="gpu device ids for CUDA_VISIBLE_DEVICES"
)
parser.add_argument("--seed", default=0, type=int, metavar="t", help="random seed")
parser.add_argument("--test-only", action="store_true", help="test only")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
set_random_seed(args.seed)
args.image_size = [args.img_h, args.img_w]
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = args.dataset
assert dataset in ["sysu", "regdb"], "dataset must be sysu or regdb"
if dataset == "sysu":
    args.dataset_path = os.path.join(args.dataset_root, "SYSU")
    class_num = 395
elif dataset == "regdb":
    args.dataset_path = os.path.join(args.dataset_root, "RegDB")
    class_num = 395

dataloaders = Loaders(args)
net = embed_net(class_num, "off", args.pooling_type)  # off means without nonlocal

# build loss
criterion = Criterion(
    [
        {"criterion": CrossEntropyLabelSmooth(num_classes=class_num), "weight": 1.0,},
        {"criterion": TripletLoss(margin="soft", metric="euclidean"), "weight": 1.0,},
    ]
)

# build optimizer
if args.optim == "sgd":
    args.lr = 0.1
    ignored_params = list(map(id, net.bottleneck.parameters())) + list(
        map(id, net.classifier.parameters())
    )
    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
    optimizer = torch.optim.SGD(
        [
            {"params": base_params, "lr": 0.1 * args.lr},
            {"params": net.bottleneck.parameters(), "lr": args.lr},
            {"params": net.classifier.parameters(), "lr": args.lr},
        ],
        weight_decay=5e-4,
        momentum=0.9,
        nesterov=True,
    )
else:
    args.lr = 0.00035
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)

lr_scheduler = WarmupMultiStepLR(
    optimizer, milestones=[40, 70], gamma=0.1, warmup_factor=0.01, warmup_epochs=10
)
optimizer = Optimizer(optimizer=optimizer, lr_scheduler=lr_scheduler, max_epochs=120)

args.results_dir = os.path.join(
    args.results_dir,
    dataset,
    "{}_pooling_type_{}".format(args.optim, args.pooling_type),
)
# run
solver = Engine(
    results_dir=args.results_dir,
    dataloaders=dataloaders,
    model=net,
    criterion=criterion,
    optimizer=optimizer,
    use_gpu=True,
    seed=args.seed,
    test_dataset=args.test_dataset,
    test_mode=args.test_mode,
)
# train
solver.train(eval_freq=10)
# test
solver.resume_latest_model()
solver.eval(args.test_dataset)
solver.eval2(args.test_dataset)
