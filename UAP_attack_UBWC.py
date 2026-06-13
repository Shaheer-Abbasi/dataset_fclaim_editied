
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
import subprocess
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

from scipy.stats import beta
from PIL import Image
from torchvision.models import resnet18
from numpy import asarray
import pickle
from scipy import stats
import math
from collections import deque
import open_clip

from scipy.stats import ttest_rel
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


def robust_rmtree(path):
    if not os.path.exists(path):
        return
    # Use system rm which handles this correctly on NFS/shared filesystems
    result = subprocess.run(['rm', '-rf', str(path)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to delete {path}: {result.stderr}")

from noise import pnoise2

####################################################################################################
### data marking 2):  injecting procedural noise (adapted from Co et al. in CCS'19, https://github.com/kenny-co/procedural-advml )
####################################################################################################
def perlin(size, period, octave, freq_sine, lacunarity = 2, base =0):
    def normalize(vec):
        vmax = np.amax(vec)
        vmin  = np.amin(vec)
        return (vec - vmin) / (vmax - vmin)
    noise = np.empty((size[1], size[2]), dtype = np.float32)
    for x in range(size[1]):
        for y in range(size[2]):
            noise[x][y] = pnoise2(x / period, y / period, octaves = octave, lacunarity = lacunarity, base=base)  
    # Sine function color map
    noise = normalize(noise)
    noise = np.sin(noise * freq_sine * np.pi)
    return normalize(noise)


def colorize(img, color = [1, 1, 1]):
    ### Visualize Image ###
    '''
    Color image

    img              has dimension 2 or 3, pixel range [0, 1]
    color            is [a, b, c] where a, b, c are from {-1, 0, 1}
    '''
    if img.ndim == 2: # expand to include color channels
        img = np.expand_dims(img, 2)
        img = np.repeat(img, 3, axis=2)  # replicate grayscale to RGB: (H,W,1) -> (H,W,3)
    color = np.array(color).reshape(1, 1, 3)  # reshape for proper broadcasting
    return (img - 0.5) * color + 0.5 # output pixel range [0, 1]


def generate_noise_image(image_shape):

    seed = np.random.randint(0, 2**32 - 1)
    rstate = np.random.RandomState(seed)

    # period = rstate.randint(30, 60)
    # octave = rstate.randint(1, 5)
    # freq_sine = rstate.randint(20, 60)

    # Optimal ranges for CIFAR-100 / CIFAR-10 (32x32 resolution)
    period = rstate.randint(2, 6)        # Micro base grid blocks (2x2 up to 6x6 pixel cells)
    octave = rstate.randint(3, 5)        # 3 to 4 octaves max (higher risks washing out into uniform gray)
    freq_sine = rstate.randint(60, 100)  # Ultra-high sine frequencies for extreme ring densit

    noise = perlin(
        size=image_shape,
        period=period,
        octave=octave,
        freq_sine=freq_sine
    )

    noise = colorize(noise)  # expected shape (H,W,3)

    noise_uint8 = (np.clip(noise, 0, 1) * 255).astype(np.uint8)

    return Image.fromarray(noise_uint8)


def _load_surrogate_and_source_images(attack_list, args):
    """Each attack needs source class, CLIP surrogate, and source images"""
    attack_list_dir = {}
    for i, j in attack_list:
        attack_list_dir[i] = attack_list_dir.get(i, 0) + 1
    source_class = max(attack_list_dir, key=attack_list_dir.get)
    print(f'Source class: {source_class}')

    base_model, _, _ = open_clip.create_model_and_transforms(
        'ViT-B-16', pretrained='datacomp_xl_s13b_b90k',
        device='cuda:0', cache_dir='./data/model/'
    )
    clip_encode = base_model.encode_image
    for p in base_model.parameters():
        p.requires_grad = False
    clip_encode.eval()

    processor = transforms.Compose([
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
        transforms.Resize(224),
    ])
    to_tensor = transforms.Compose([transforms.ToTensor()])

    source_images = []
    for cls, fname in attack_list:
        if cls == source_class:
            img = to_tensor(Image.open(args.img_path + cls + '/' + fname)).unsqueeze(0).cuda()
            source_images.append(img)
    source_images = torch.cat(source_images, dim=0)

    return source_class, clip_encode, processor, to_tensor, source_images


def attack_PGD(attack_list, args):
    """PGD optimization-based UAP attack"""

    source_class, clip_encode, processor, _, source_images = \
        _load_surrogate_and_source_images(attack_list, args)

    alpha = 0.2
    n_iters = getattr(args, 'uap_iters', 200)
    lr = getattr(args, 'uap_lr', 0.01)
    batch_size = min(32, len(source_images))

    with torch.no_grad():
        all_clean_feat = []
        for start in range(0, len(source_images), batch_size):
            batch = source_images[start:start + batch_size]
            feat = clip_encode(processor(batch))
            all_clean_feat.append(F.normalize(feat, dim=-1))
        all_clean_feat = torch.cat(all_clean_feat, dim=0)

    best_trigger = None
    best_loss = float('inf')

    for restart in range(getattr(args, 'uap_restarts', 3)):
        delta = torch.rand(1, 3, 32, 32, device='cuda')
        delta = Variable(delta, requires_grad=True)
        optimizer = optim.Adam([delta], lr=lr)

        for iteration in range(n_iters):
            indices = torch.randperm(len(source_images))[:batch_size]
            batch = source_images[indices]
            clean_feat = all_clean_feat[indices]

            optimizer.zero_grad()

            delta_clamped = delta.clamp(0, 1)
            perturbed = ((1 - alpha) * batch + alpha * delta_clamped).clamp(0, 1)
            perturbed_feat = clip_encode(processor(perturbed))
            perturbed_feat = F.normalize(perturbed_feat, dim=-1)

            cos_sim = (clean_feat * perturbed_feat).sum(dim=-1).mean()
            cos_sim.backward()
            optimizer.step()

            if iteration % 50 == 0:
                print(f'  [restart {restart}] PGD iter {iteration}/{n_iters}, '
                      f'cos_sim: {cos_sim.item():.4f}')

        # Evaluate this restart on the full set
        with torch.no_grad():
            delta_clamped = delta.clamp(0, 1)
            total_cos = 0.0
            for start in range(0, len(source_images), batch_size):
                batch = source_images[start:start + batch_size]
                feat_c = all_clean_feat[start:start + batch_size]
                pert = ((1 - alpha) * batch + alpha * delta_clamped).clamp(0, 1)
                feat_p = F.normalize(clip_encode(processor(pert)), dim=-1)
                total_cos += (feat_c * feat_p).sum(dim=-1).sum().item()
            avg_cos = total_cos / len(source_images)

        if avg_cos < best_loss:
            best_loss = avg_cos
            best_trigger = delta.detach().clamp(0, 1).clone()
            print(f'  [restart {restart}] New best avg cos_sim: {avg_cos:.4f}')

    trigger_np = best_trigger.squeeze(0).permute(1, 2, 0).cpu().numpy()
    trigger_np = (trigger_np * 255).astype(np.uint8)
    trigger = Image.fromarray(trigger_np)

    return source_class, trigger


def _build_patch_trigger_and_mask(patch_img, patch_size, img_size=32):
    """Place perlin patch at bottom-right of image"""
    mask = np.zeros((img_size, img_size, 3), dtype=np.float32)
    trigger = np.zeros((img_size, img_size, 3), dtype=np.float32)
    r0, c0 = img_size - patch_size, img_size - patch_size
    patch_np = np.array(patch_img).astype(np.float32)
    trigger[r0:img_size, c0:img_size, :] = patch_np
    mask[r0:img_size, c0:img_size, :] = 1.0
    return trigger, mask


def attack_perlin(attack_list, args):
    """Perlin patch UAP attack"""

    source_class, clip_encode, processor, to_tensor, source_images = \
        _load_surrogate_and_source_images(attack_list, args)

    alpha = 0.2
    batch_size = min(32, len(source_images))
    img_size = 32
    patch_size = getattr(args, 'patch_size', 8)
    n_candidates = getattr(args, 'uap_perlin_candidates', 100)

    print(f'  Patch size: {patch_size}x{patch_size} at bottom-right corner')

    noises = [
        generate_noise_image((3, patch_size, patch_size))
        for _ in range(n_candidates)
    ]

    mask_np = np.zeros((img_size, img_size, 3), dtype=np.float32)
    r0, c0 = img_size - patch_size, img_size - patch_size
    mask_np[r0:img_size, c0:img_size, :] = 1.0
    mask_t = to_tensor(
        Image.fromarray((mask_np * 255).astype(np.uint8))
    ).unsqueeze(0).cuda()

    max_distance = 0
    best_trigger_np = None
    best_mask_np = None

    for noise_img in noises:
        trigger_np, candidate_mask = _build_patch_trigger_and_mask(
            noise_img, patch_size, img_size
        )
        trigger_t = to_tensor(
            Image.fromarray(trigger_np.astype(np.uint8))
        ).unsqueeze(0).cuda()

        dis = 0.0
        with torch.no_grad():
            for start in range(0, len(source_images), batch_size):
                batch = source_images[start:start + batch_size]
                clean_feat = clip_encode(processor(batch))
                perturbed = (
                    (1 - alpha * mask_t) * batch + alpha * mask_t * trigger_t
                ).clamp(0, 1)
                pert_feat = clip_encode(processor(perturbed))
                dis += torch.norm(clean_feat - pert_feat, dim=-1).sum().item()

        if dis > max_distance:
            max_distance = dis
            best_trigger_np = trigger_np
            best_mask_np = candidate_mask

    print(f'  Best Perlin patch displacement: {max_distance:.4f}')

    trigger = Image.fromarray(best_trigger_np.astype(np.uint8))
    return source_class, trigger, best_mask_np


def create_backdoor_testset(trigger, source_class, args, mask=None):

    shutil.copytree(args.test_path, args.backdoored_test_path, dirs_exist_ok=True) 

    trigger = np.array(trigger).astype(np.float32)
    alpha = 0.2

    for j in os.listdir(args.backdoored_test_path + source_class):
        img = Image.open(args.backdoored_test_path + source_class + '/' + j)
      
        img = np.array(img).astype(np.float32)
        if mask is None:
            marked_image = (1 - alpha) * img + alpha * trigger
        else:
            marked_image = (1 - alpha * mask) * img + alpha * mask * trigger
        marked_image = np.clip(marked_image, 0, 255).astype(np.uint8)
        marked_image = Image.fromarray(marked_image)
        marked_image.save(args.backdoored_test_path + '/' + source_class + '/' + j)
    print('backdoored testset created.')


def test(testloader, model, use_cuda=True):
    model.eval()
    return_output = []
    for _, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        with torch.no_grad():
            outputs = torch.nn.functional.softmax(model(inputs), dim=1)[0][targets[0]]
            return_output.append(outputs.cpu().detach().numpy())

    return np.array(return_output)


def UWBC_test(trigger, SOURCE_CLASS, args, mask=None):

    # load model
    ckpt = torch.load(args.target_path + 'target_model.pth')
    target_model = mymodels.get_model(args.net[0], args.dataset, args.pretrained)
    target_model.cuda()
    target_model.load_state_dict({k.replace("module.", ""): v for k, v in ckpt.items()}, strict=True)
    target_model.eval()
    print('loaded model')

    args.data_mean = data_mean
    args.data_std = data_std

    # load dataset
    clean_dataset = torchvision.datasets.ImageFolder(root=args.test_path, transform=transforms.ToTensor())
    clean_dataset.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                            torchvision.transforms.Normalize(args.data_mean, args.data_std)])

    create_backdoor_testset(trigger, SOURCE_CLASS, args, mask=mask)

    posioned_dataset = torchvision.datasets.ImageFolder(root=args.backdoored_test_path, transform=transforms.ToTensor())
    posioned_dataset.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                            torchvision.transforms.Normalize(args.data_mean, args.data_std)])

    clean_data_loader = torch.utils.data.DataLoader(clean_dataset, batch_size=1, drop_last=False, shuffle=False)

    # keep test data that are correctly classified by the target model
    kept_idx = []
    for idx, (inputs, targets) in enumerate(clean_data_loader):
        if args.classes[targets[0]] == SOURCE_CLASS:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = target_model(inputs)
            outputs = torch.argmax(outputs, dim=1)[0]
            if outputs == targets[0]:
                kept_idx.append(idx)

    print(len(kept_idx))
    random.shuffle(kept_idx)

    clean_dataset = [clean_dataset[i] for i in kept_idx]
    poisoned_dataset = [posioned_dataset[i] for i in kept_idx]

    clean_data_loader = torch.utils.data.DataLoader(clean_dataset, batch_size=1, drop_last=False, shuffle=False)
    poisoned_data_loader = torch.utils.data.DataLoader(poisoned_dataset, batch_size=1, drop_last=False, shuffle=False)

    output_clean = test(clean_data_loader, target_model) 
    output_poisoned = test(poisoned_data_loader, target_model)

    print(np.mean(output_clean))
    print(np.mean(output_poisoned))

    T_test = ttest_rel(output_poisoned + 0.2, output_clean, alternative='less')

    print(T_test)

    if T_test[1] < 0.05:
        detection = 1
    else:
        detection = 0

    return detection


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
    parser.add_argument('--data_path', default='./data/data', type=str)
    parser.add_argument('--img_path', type=str, default='./data/cifar100/train/')
    parser.add_argument('--published_path', type=str, default='./data/cifar100/')
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
    parser.add_argument("--test_path", type=str, default='./data/cifar100/test/')
    parser.add_argument("--backdoored_test_path", type=str, default='./data/cifar100/')

    # Debugging:
    parser.add_argument('--dryrun', action='store_true')
    parser.add_argument('--save', default='full', help='Export poisons into a given format. Options are full/limited/automl/numpy.')

    # Distributed Computations
    parser.add_argument("--local_rank", default=None, type=int, help='Distributed rank. This is an INTERNAL ARGUMENT! '
                                                                     'Only the launch utility should set this argument!')
    
    parser.add_argument("--exp_index", default=0, type=int)

    # UAP attack options
    parser.add_argument("--attack_method", default="pgd", type=str,
                        choices=["pgd", "perlin_uap"],
                        help="UAP attack variant: 'pgd' for full-image optimization, "
                             "'perlin_uap' for localized Perlin patch at fixed corner")
    parser.add_argument("--uap_iters", default=200, type=int,
                        help="Number of PGD optimization iterations (pgd method)")
    parser.add_argument("--uap_lr", default=0.01, type=float,
                        help="Learning rate for PGD optimization")
    parser.add_argument("--uap_restarts", default=3, type=int,
                        help="Number of random restarts for PGD (pgd method)")
    parser.add_argument("--uap_perlin_candidates", default=100, type=int,
                        help="Number of Perlin patch candidates to screen (perlin_uap method)")
    parser.add_argument("--patch_size", default=8, type=int,
                        help="Side length of localized Perlin patch (perlin_uap method)")

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
    
    args.target_path = './data/cifar100/target({})/'.format(args.exp_index)

    total_samples = 25000
    number_per_class = 250

    args.all_img_list = []
    listing = args.classes
    for i in listing:

        file_list1 = os.listdir(args.img_path + i)
        file_list2 = os.listdir(args.target_path + i)

        for j in file_list1:
            if j not in file_list2:
                args.all_img_list.append((i, j))


    for class_imbalance in [2, 3, 4]:
        for list_size in [1000, 2000, 5000, 10000]:
            
            if True:
            # if not os.path.exists('./UBWC/results/false_detection_attack(no_model)({})({})({}).pickle'.format(class_imbalance, list_size, args.exp_index)):
                print('================= class_imbalance: {} | list_size: {} ==================='.format(class_imbalance, list_size))

                results_all = 0

                for _ in range(20):

                    if class_imbalance == 1:
                        attack_list = random.sample(args.all_img_list, list_size)
                    else:
                        # Define the point at which you want to evaluate the PDF
                        old = 0
                        # total_num = 0
                        attack_list = []
                        classes = args.classes.copy()
                        random.shuffle(classes)
                        for j in range(1, len(classes)+1):
                            cdf_value = beta.cdf(j/len(classes), class_imbalance, class_imbalance)
                            sample_list1 = os.listdir(args.img_path + classes[j-1])
                            sample_list2 = os.listdir(args.target_path + classes[j-1])
                            sample_list = [z for z in sample_list1 if z not in sample_list2]
                            attack_list_ = random.sample(sample_list, min(int((cdf_value-old)*list_size) + 1, number_per_class))
                            attack_list__ = [(classes[j-1], z) for z in attack_list_]
                            attack_list += attack_list__
                            # args.num_per_class[args.classes[j-1]] = min(int((cdf_value-old)*list_size) + 1, number_per_class)
                            # total_num += min(int((cdf_value-old)*list_size) + 1, number_per_class)
                            old = cdf_value

                    # 'design' images that are used to falsely claim
                    if args.attack_method == 'pgd':
                        source_class, trigger = attack_PGD(attack_list, args)
                        results = UWBC_test(trigger, source_class, args)
                    elif args.attack_method == 'perlin_uap':
                        source_class, trigger, mask = attack_perlin(attack_list, args)
                        results = UWBC_test(trigger, source_class, args, mask=mask)

                    results_all += results

                print(results_all)

    print('-------------Job finished.-------------------------')
