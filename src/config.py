import os

from easydict import EasyDict as edict

import torch
import numpy as np

from utils.transforms import *


# Dataset path
PATH = edict()

PATH.DB_ROOT = '../data/kaist-rgbt/'
PATH.JSON_GT_FILE = os.path.join('kaist_annotations_test20.json' )

# train
train = edict()

train.day = "all"
train.img_set = f"train-all-20.txt"

train.checkpoint = None ## Load chekpoint

train.batch_size = 6 # batch size

train.start_epoch = 0  # start at this epoch
train.epochs = 30  # number of epochs to run without early-stopping
train.epochs_since_improvement = 3  # number of epochs since there was an improvement in the validation metric
train.best_loss = 100.  # assume a high loss at first

train.lr = 1e-4   # learning rate
train.momentum = 0.9  # momentum
train.weight_decay = 5e-4  # weight decay
train.grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

train.print_freq = 10

train.annotation = "AR-CNN" # AR-CNN, Sanitize, Original


train.load_model = "" # if you want to tune regressor of previous model: "jobs/model_name/checkpoint_ssd300.pth.tar029"
train.tune_regressor_only = False

# test & eval
test = edict()

test.result_path = './result' ### coco tool. Save Results(jpg & json) Path

test.day = "all" # all, day, night
test.img_set = f"test-{test.day}-20.txt"

test.annotation = "AR-CNN"

test.input_size = [512., 640.]

### test model ~ eval.py
test.checkpoint = "./jobs/best_checkpoint.pth.tar"
test.batch_size = 4

### train_eval.py
test.eval_batch_size = 1


# KAIST Image Mean & STD
## RGB
IMAGE_MEAN = [0.3465,  0.3219,  0.2842]
IMAGE_STD = [0.2358, 0.2265, 0.2274]
## Lwir
LWIR_MEAN = [0.1598]
LWIR_STD = [0.0813]

                    
# dataset
dataset = edict()
dataset.workers = 8
dataset.OBJ_LOAD_CONDITIONS = {    
                                  'train': {'hRng': (12, np.inf), 'xRng':(5, 635), 'yRng':(5, 507), 'wRng':(-np.inf, np.inf)}, 
                                  'test': {'hRng': (-np.inf, np.inf), 'xRng':(5, 635), 'yRng':(5, 507), 'wRng':(-np.inf, np.inf)}, 
                              }

# main
args = edict(path=PATH,
             train=train,
             test=test,
             dataset=dataset)

args.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

args.exp_time = None
args.exp_name = None

args.n_classes = 3

## Semi Unpaired Augmentation
args.unpaired_augmentation = ["TT_RandomHorizontalFlip",
                             "TT_RandomHorizontalFlip_fixed",
                             "TT_FixedHorizontalFlip",
                             "TT_RandomResizedCrop"]
## Train dataset transform                             
args["train"].img_transform = Compose([ ColorJitter(0.3, 0.3, 0.3), 
                                        ColorJitterLWIR(contrast=0.3) ])

args["train"].co_transform = Compose([  TT_RandomHorizontalFlip_fixed(flip_vis_prob=0.5, flip_lwir_prob=0.5),
                                        TT_RandomResizedCrop([512,640], \
                                                                scale=(0.25, 4.0), \
                                                                ratio=(0.8, 1.2)),
                                        ToTensor(), \
                                        Normalize(IMAGE_MEAN, IMAGE_STD, 'R'), \
                                        Normalize(LWIR_MEAN, LWIR_STD, 'T') ], \
                                        args=args)

# if you want to tune regressor of previous model:
# args["train"].co_transform = Compose([  RandomHorizontalShift(p=1.0),
#                                         TT_RandomHorizontalFlip_fixed(flip_vis_prob=0.5, flip_lwir_prob=0.5),
#                                         TT_RandomResizedCrop([512,640], \
#                                                                 scale=(0.25, 4.0), \
#                                                                 ratio=(0.8, 1.2)),
#                                         ToTensor(), \
#                                         Normalize(IMAGE_MEAN, IMAGE_STD, 'R'), \
#                                         Normalize(LWIR_MEAN, LWIR_STD, 'T') ], \
#                                         args=args)

## Test dataset transform
args["test"].img_transform = Compose([ ])    
args["test"].co_transform = Compose([Resize(test.input_size), \
                                     ToTensor(), \
                                     Normalize(IMAGE_MEAN, IMAGE_STD, 'R'), \
                                     Normalize(LWIR_MEAN, LWIR_STD, 'T')                        
                                    ])