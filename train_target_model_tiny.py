
import torch

from torchvision import transforms
from torch import nn
import torchvision
import argparse
import numpy as np
import os
import shutil
from PIL import Image
import warnings
import forest
from PIL import Image
import mymodels


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


def get_parser():
    """Construct the central argument parser, filled with useful defaults.

    The first block is essential to test poisoning in different scenarios.
    The options following afterwards change the algorithm in various ways and are set to reasonable defaults.
    """
    parser = argparse.ArgumentParser(description='Construct poisoned training data for the given network and dataset')

    ###########################################################################
    # Central:
    parser.add_argument('--net', default='ResNet18', type=lambda s: [str(item) for item in s.split(',')])
    parser.add_argument('--dataset', default='TinyImageNet', type=str, choices=['CIFAR10', 'CIFAR100', 'ImageNet', 'ImageNet1k', 'MNIST', 'TinyImageNet'])
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
    parser.add_argument('--data_path', default='./data/data/tiny-imagenet-200/', type=str)
    parser.add_argument('--img_path', type=str, default='./data/tiny/train/')
    parser.add_argument('--published_path', type=str, default='./data/tiny/')
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
    parser.add_argument("--test_path", type=str, default='./data/tiny/test/')
    parser.add_argument("--marked_test_path", type=str, default='./data/tiny/marked_test/')

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


    setup = forest.utils.system_startup(args)

    model = forest.Victim(args, setup=setup)
    data = forest.Kettle(args, model.defs.batch_size, model.defs.augmentations, setup=setup)
    data_mean, data_std = data.trainset.data_mean, data.trainset.data_std

    args.image_mean = data_mean
    args.image_std = data_std
    args.classes = data.trainset.classes
    args.data_transform = data.trainset.transform
    args.data_augmentation = data.augment

    args.target_path = './data/tiny/target({})/'.format(args.exp_index)
    
    # train target model
    model = forest.Victim(args, setup=setup)
    data = forest.Kettle(args, model.defs.batch_size, model.defs.augmentations, setup=setup)
    data_mean, data_std = data.trainset.data_mean, data.trainset.data_std

    print('Training a {} model on published data...'.format(args.net[0]))
    prepare_data_loader(args.target_path, data, model, data_mean, data_std, args)
    stats_results = model.validate(data, 1)
    print('Saving target model to {}...'.format(args.target_path + 'target_model.pth'))
    torch.save(model.model.state_dict(), args.target_path + 'target_model.pth')


    print('-------------Job finished.-------------------------')
