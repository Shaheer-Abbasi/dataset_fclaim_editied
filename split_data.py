import comet_ml
from comet_ml import Experiment

import torch
from torchvision import models
from torchvision import transforms
from torch import nn
import torchvision
from torch.nn import functional as F
from torch import optim, autograd
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.autograd import Variable

import datetime
import time
import argparse
import numpy as np
import os
import shutil
import json
from os.path import basename, join
from PIL import Image
from numpy import asarray
from copy import deepcopy
import warnings
import matplotlib.pyplot as plt

import forest

from src.model import build_model
from src.stats import cosine_pvalue
from src.dataset import getCifarTransform, NORMALIZE_CIFAR
from src.data_augmentations import RandomResizedCropFlip, CenterCrop
from src.datasets.folder import default_loader
from src.utils import bool_flag, get_optimizer, repeat_to
import random

import clip
from PIL import Image
from torchvision.models import resnet18
from numpy import asarray
import pickle
from scipy import stats
import math
from collections import deque

from scipy.stats import ttest_rel
import mymodels

from utils_ import add_noise_to_sample, gen_unique_outlier_pattern


# torch.backends.cudnn.benchmark = forest.consts.BENCHMARK
torch.multiprocessing.set_sharing_strategy(forest.consts.SHARING_STRATEGY)


torch.set_num_threads(1)

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
warnings.filterwarnings("ignore", "Metadata Warning, tag [0-9]+ had too many entries", UserWarning)


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC



def prepare_data_loader(path, data, model, data_mean, data_std, args):

    # replace the trainset with the target one
    trainset = torchvision.datasets.ImageFolder(root=path, transform=transforms.ToTensor())

    trainset.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(data_mean, data_std)])
    data.trainloader = torch.utils.data.DataLoader(trainset, batch_size=min(model.defs.batch_size, len(trainset)),
                                                    shuffle=True, drop_last=False, num_workers=4, pin_memory=True)

    print('OK')


def mark(args):
    listing = os.listdir(args.img_path)
    for i in args.classes:
        # if os.path.isfile(args.img_path + i):
        #     listing.remove(i)

        if os.path.exists(args.published_path + i):
            shutil.rmtree(args.published_path + i)
        if not os.path.exists(args.published_path + i):
            os.mkdir(args.published_path + i)
   
    if int(args.mark_budget * 50000) < 100:
        marked_class = random.sample(args.classes, int(args.mark_budget * 50000))
        per_class = 1
    else:
        marked_class = args.classes
        per_class = int(args.mark_budget * 50000 / len(marked_class))
    for i in args.classes:
        if i in marked_class:
            args.marked_file[i] = random.sample(os.listdir(args.img_path + i), per_class)
        else:
            args.marked_file[i] = []

    sample_as_random_seed = np.random.randint(0, 256, size=(3, 32, 32), dtype=np.uint8)
    mark_pattern = gen_unique_outlier_pattern(sample_as_random_seed)[0] * 255.0  # convert [0,1] to [0,255] and extract from list
    mark_pattern = np.transpose(mark_pattern, (1, 2, 0)).astype(np.float32)  # CHW to HWC

    for i in args.classes:
        file_list1 = os.listdir(args.img_path + i)

        for j in range(len(file_list1)):

            image = Image.open(args.img_path + i + '/' + file_list1[j]).convert("RGB")

            if i in args.marked_file.keys() and file_list1[j] in args.marked_file[i]:

                noisy_image = add_noise_to_sample(image)
                noisy_array = np.array(noisy_image).astype(np.float32)

                marked_image = (1-args.alpha) * noisy_array + args.alpha * mark_pattern

                marked_image = np.clip(marked_image, 0, 255).astype(np.uint8)
                marked_image = Image.fromarray(marked_image)
                marked_image.save(args.published_path + '/' + i + '/' + file_list1[j])
            else:
                image.save(args.published_path + '/' + i + '/' + file_list1[j])
    print('finished marking.')

    # if args.exp_index == 0 and not os.path.exists('/usr/project/xtmp/zh127/cifar100/marked_test/'):
    if True:

        args.marked_test_path = '/usr/project/xtmp/zh127/cifar100/marked_test({})/'.format(args.exp_index)
        if not os.path.exists(args.marked_test_path):
            os.mkdir(args.marked_test_path)

        for i in args.classes:

            if os.path.exists(args.marked_test_path + i):
                shutil.rmtree(args.marked_test_path + i)
            if not os.path.exists(args.marked_test_path + i):
                os.mkdir(args.marked_test_path + i)
    
        sample_as_random_seed = np.random.randint(0, 256, size=(3, 32, 32), dtype=np.uint8)
        mark_pattern = gen_unique_outlier_pattern(sample_as_random_seed)[0] * 255.0  # convert [0,1] to [0,255] and extract from list
        mark_pattern = np.transpose(mark_pattern, (1, 2, 0)).astype(np.float32)  # CHW to HWC

        for i in args.classes:
            file_list1 = os.listdir(args.test_path + i)

            for j in range(len(file_list1)):

                image = Image.open(args.test_path + i + '/' + file_list1[j]).convert("RGB")

                noisy_image = add_noise_to_sample(image)
                noisy_array = np.array(noisy_image).astype(np.float32)

                marked_image = (1-args.alpha) * noisy_array + args.alpha * mark_pattern

                marked_image = np.clip(marked_image, 0, 255).astype(np.uint8)
                marked_image = Image.fromarray(marked_image)
                marked_image.save(args.marked_test_path + '/' + i + '/' + file_list1[j])

    print('finished marking.')


def test(transform, args):

    print(transform)

    # load model
    ckpt = torch.load(args.published_path + 'target_model.pth')
    target_model = mymodels.get_model(args.net[0], args.dataset, args.pretrained)
    target_model.cuda()
    target_model.load_state_dict({k.replace("module.", ""): v for k, v in ckpt.items()}, strict=True)
    target_model.eval()
    print('loaded model')

    number_samples = int(args.mark_budget * 50000 / 100)

    marked_test_pool = []
    for i in args.classes:
        file_list1 = os.listdir(args.marked_test_path + i)
        for j in range(len(file_list1)):
            marked_test_pool.append((i, file_list1[j]))

    minimal_marked_test_loss = 100

    criterion = nn.CrossEntropyLoss(reduction='sum')

    for k in range(5000):

        test_samples = random.sample(marked_test_pool, number_samples)

        batches = number_samples // 256 + 1
        average_loss = 0
        for batches_idx in range(batches):
            
            if batches_idx == batches - 1:
                test_samples_ = [transform(Image.open(args.marked_test_path + i + '/' + j)).unsqueeze(0).cuda() for i, j in test_samples[batches_idx*256:]]
                labels = torch.tensor([args.classes.index(i) for i, j in test_samples[batches_idx*256:]]).cuda()
            else:
                test_samples_ = [transform(Image.open(args.marked_test_path + i + '/' + j)).unsqueeze(0).cuda() for i, j in test_samples[batches_idx*256:(batches_idx+1)*256]]
                labels = torch.tensor([args.classes.index(i) for i, j in test_samples[batches_idx*256:(batches_idx+1)*256]]).cuda()
            test_samples_ = torch.cat(test_samples_, dim=0)

            with torch.no_grad():
                logits = target_model(test_samples_)
                loss = criterion(logits, labels)

            average_loss += loss.item()
        average_loss = average_loss / number_samples
        if average_loss < minimal_marked_test_loss:
            minimal_marked_test_loss = average_loss
    print(minimal_marked_test_loss)
    
    marked_pool = []
    for i in args.classes:
        marked_pool += [(i, j) for j in args.marked_file[i]]

    batches = number_samples // 256 + 1
    average_loss = 0
    for batches_idx in range(batches):
            
        if batches_idx == batches - 1:
            test_samples = [transform(Image.open(args.published_path + i + '/' + j)).unsqueeze(0).cuda() for i, j in marked_pool[batches_idx*256:]]
            labels = torch.tensor([args.classes.index(i) for i, j in marked_pool[batches_idx*256:]]).cuda()
        else:
            test_samples = [transform(Image.open(args.published_path + i + '/' + j)).unsqueeze(0).cuda() for i, j in marked_pool[batches_idx*256:(batches_idx+1)*256]]
            labels = torch.tensor([args.classes.index(i) for i, j in marked_pool[batches_idx*256:(batches_idx+1)*256]]).cuda()
        test_samples = torch.cat(test_samples, dim=0)

        with torch.no_grad():
            logits = target_model(test_samples)
            loss = criterion(logits, labels)

        average_loss += loss.item()
    average_loss = average_loss / number_samples
    print(average_loss)
    if average_loss < minimal_marked_test_loss:
        return 1
    else:
        return 0


def get_parser():
    """Construct the central argument parser, filled with useful defaults.

    The first block is essential to test poisoning in different scenarios.
    The options following afterwards change the algorithm in various ways and are set to reasonable defaults.
    """
    parser = argparse.ArgumentParser(description='Construct poisoned training data for the given network and dataset')

    ###########################################################################
    # Central:
    parser.add_argument('--net', default='ResNet18', type=lambda s: [str(item) for item in s.split(',')])
    parser.add_argument('--dataset', default='CIFAR100', type=str, choices=['CIFAR10', 'CIFAR100', 'ImageNet', 'ImageNet1k', 'MNIST', 'TinyImageNet'])
    parser.add_argument('--recipe', default='gradient-matching', type=str, choices=['gradient-matching', 'gradient-matching-private',
                                                                                    'watermarking', 'poison-frogs', 'metapoison', 'bullseye'])
    parser.add_argument('--threatmodel', default='single-class', type=str, choices=['single-class', 'third-party', 'random-subset'])

    # Reproducibility management:
    parser.add_argument('--poisonkey', default=None, type=str, help='Initialize poison setup with this key.')  # Also takes a triplet 0-3-1
    parser.add_argument('--modelkey', default=None, type=int, help='Initialize the model with this key.')
    parser.add_argument('--deterministic', action='store_true', help='Disable CUDNN non-determinism.')

    # Poison properties / controlling the strength of the attack:
    parser.add_argument('--eps', default=16, type=float)
    parser.add_argument('--budget', default=0.01, type=float, help='Fraction of training data that is poisoned')
    parser.add_argument('--targets', default=1, type=int, help='Number of targets')

    # Files and folders
    parser.add_argument('--name', default='', type=str, help='Name tag for the result table and possibly for export folders.')
    parser.add_argument('--table_path', default='brew_poison/tables/', type=str)
    parser.add_argument('--data_path', default='/usr/project/xtmp/zh127/data', type=str)
    parser.add_argument('--img_path', type=str, default='/usr/project/xtmp/zh127/cifar100/train/')
    parser.add_argument('--published_path', type=str, default='/usr/project/xtmp/zh127/cifar100/')
    ###########################################################################

    # Poison brewing:
    parser.add_argument('--attackoptim', default='signAdam', type=str)
    parser.add_argument('--attackiter', default=250, type=int)
    parser.add_argument('--init', default='randn', type=str)  # randn / rand
    parser.add_argument('--tau', default=0.1, type=float)
    parser.add_argument('--scheduling', action='store_false', help='Disable step size decay.')
    parser.add_argument('--target_criterion', default='cross-entropy', type=str, help='Loss criterion for target loss')
    parser.add_argument('--restarts', default=8, type=int, help='How often to restart the attack.')

    parser.add_argument('--pbatch', default=512, type=int, help='Poison batch size during optimization')
    parser.add_argument('--pshuffle', action='store_true', help='Shuffle poison batch during optimization')
    parser.add_argument('--paugment', action='store_false', help='Do not augment poison batch during optimization')
    parser.add_argument('--data_aug', type=str, default='default', help='Mode of diff. data augmentation.')

    # Poisoning algorithm changes
    parser.add_argument('--full_data', action='store_true', help='Use full train data (instead of just the poison images)')
    parser.add_argument('--adversarial', default=0, type=float, help='Adversarial PGD for poisoning.')
    parser.add_argument('--ensemble', default=1, type=int, help='Ensemble of networks to brew the poison on')
    parser.add_argument('--stagger', action='store_true', help='Stagger the network ensemble if it exists')
    parser.add_argument('--step', action='store_true', help='Optimize the model for one epoch.')
    parser.add_argument('--max_epoch', default=None, type=int, help='Train only up to this epoch before poisoning.')

    # Use only a subset of the dataset:
    parser.add_argument('--ablation', default=1.0, type=float, help='What percent of data (including poisons) to use for validation')

    # Gradient Matching - Specific Options
    parser.add_argument('--loss', default='similarity', type=str)  # similarity is stronger in  difficult situations

    # These are additional regularization terms for gradient matching. We do not use them, but it is possible
    # that scenarios exist in which additional regularization of the poisoned data is useful.
    parser.add_argument('--centreg', default=0, type=float)
    parser.add_argument('--normreg', default=0, type=float)
    parser.add_argument('--repel', default=0, type=float)

    # Specific Options for a metalearning recipe
    parser.add_argument('--nadapt', default=2, type=int, help='Meta unrolling steps')
    parser.add_argument('--clean_grad', action='store_true', help='Compute the first-order poison gradient.')

    # Validation behavior
    parser.add_argument('--vruns', default=1, type=int, help='How often to re-initialize and check target after retraining')
    parser.add_argument('--vnet', default=None, type=lambda s: [str(item) for item in s.split(',')], help='Evaluate poison on this victim model. Defaults to --net')
    parser.add_argument('--retrain_from_init', action='store_true', help='Additionally evaluate by retraining on the same model initialization.')

    # Optimization setup
    parser.add_argument('--pretrained', action='store_true', help='Load pretrained models from torchvision, if possible [only valid for ImageNet].')
    parser.add_argument('--optimization', default='conservative', type=str, help='Optimization Strategy')
    parser.add_argument('--regularization', default=None, type=float, help='Add custom gradient noise during training.')
    # Strategy overrides:
    parser.add_argument('--epochs', default=80, type=int)
    parser.add_argument('--noaugment', action='store_true', help='Do not use data augmentation during training.')
    parser.add_argument('--gradient_noise', default=None, type=float, help='Add custom gradient noise during training.')
    parser.add_argument('--gradient_clip', default=None, type=float, help='Add custom gradient clip during training.')

    # Optionally, datasets can be stored as LMDB or within RAM:
    parser.add_argument('--lmdb_path', default=None, type=str)
    parser.add_argument('--cache_dataset', action='store_true', help='Cache the entire thing :>')

    # These options allow for testing against the toxicity benchmark found at
    # https://github.com/aks2203/poisoning-benchmark
    parser.add_argument('--benchmark', default='', type=str, help='Path to benchmarking setup (pickle file)')
    parser.add_argument('--benchmark_idx', default=0, type=int, help='Index of benchmark test')

    # attack model and shadow model
    parser.add_argument('--attack_epochs', default=40, type=int)
    parser.add_argument('--lr_attack', default=0.001, type=float)
    parser.add_argument('--batch', default=256, type=int)
    parser.add_argument('--mark_budget', default=0.001, type=float)

    # mark:
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--mepochs", type=int, default=90)
    parser.add_argument("--lambda_ft_l2", type=float, default=0.01)
    parser.add_argument("--lambda_l2_img", type=float, default=0.0005)
    parser.add_argument("--moptimizer", type=str, default="sgd,lr=1.0")

    # detection
    parser.add_argument("--test_path", type=str, default='/usr/project/xtmp/zh127/cifar100/test/')
    parser.add_argument("--marked_test_path", type=str, default='/usr/project/xtmp/zh127/cifar100/marked_test/')

    # Debugging:
    parser.add_argument('--dryrun', action='store_true')
    parser.add_argument('--save', default='full', help='Export poisons into a given format. Options are full/limited/automl/numpy.')

    # Distributed Computations
    parser.add_argument("--local_rank", default=None, type=int, help='Distributed rank. This is an INTERNAL ARGUMENT! '
                                                                     'Only the launch utility should set this argument!')
    
    parser.add_argument("--exp_index", default=0, type=int)


    return parser


if __name__ == "__main__":

    # Parse input arguments
    args = get_parser().parse_args()
    # 100% reproducibility?
    if args.deterministic:
        forest.utils.set_deterministic()

    data_path = args.data_path

    # comet_ml experiment setup
    print('==> setting up comet experiment...')
    experiment = Experiment(project_name='false-detection attack', auto_param_logging=False,
                                api_key= "U1kuka6SA58EBpCa8Ct6CY1fp",
                                auto_metric_logging=False,
                                parse_args=False)
    comet_ml.config.experiment = None
    experiment.set_name('MembershipTracker ({})({})'.format(args.mark_budget, args.exp_index))
    experiment.add_tag('')
    experiment.log_parameters(vars(args))

    setup = forest.utils.system_startup(args)

    model = forest.Victim(args, setup=setup)
    data = forest.Kettle(args, model.defs.batch_size, model.defs.augmentations, setup=setup)
    data_mean, data_std = data.trainset.data_mean, data.trainset.data_std

    args.image_mean = data_mean
    args.image_std = data_std
    args.classes = data.trainset.classes
    args.data_transform = data.trainset.transform
    args.data_augmentation = data.augment
    
    for args.exp_index in range(20):

        print('============ {} ============='.format(args.exp_index))
        # data path for training target model and surrogate model
        args.target_path = './data/cifar100/MembershipTracker(target)({})/'.format(args.exp_index)
        args.surrogate_path = './data/cifar100/MembershipTracker(surrogate)({})({})({})/'.format(args.mark_budget, args.alpha, args.exp_index)

        if os.path.exists(args.target_path):
                shutil.rmtree(args.target_path)
        if not os.path.exists(args.target_path):
            os.mkdir(args.target_path)

        if os.path.exists(args.surrogate_path):
            shutil.rmtree(args.surrogate_path)
        if not os.path.exists(args.surrogate_path):
            os.mkdir(args.surrogate_path)

        for i in args.classes:

            if os.path.exists(args.target_path + i):
                shutil.rmtree(args.target_path + i)
            if not os.path.exists(args.target_path + i):
                os.mkdir(args.target_path + i)

            if os.path.exists(args.surrogate_path + i):
                shutil.rmtree(args.surrogate_path + i)
            if not os.path.exists(args.surrogate_path + i):
                os.mkdir(args.surrogate_path + i)

            target_list = random.sample(os.listdir('./data/cifar100/train/' + i), len(os.listdir('./data/cifar100/train/' + i)) // 2)
            for j in os.listdir('./data/cifar100/train/' + i):
                if j in target_list:
                    shutil.copy('./data/cifar100/train/' + i + '/' + j, args.target_path + i + '/' + j)
                else:
                    shutil.copy('./data/cifar100/train/' + i + '/' + j, args.surrogate_path + i + '/' + j)


    print('-------------Job finished.-------------------------')
