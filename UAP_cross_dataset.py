#!/usr/bin/env python3
"""Cross-dataset false-claim attack: TinyImageNet surrogate -> CIFAR-100 victim.

Supports three attack methods:
  perlin_uap_paste  -- CLIP-scored Perlin patch using TinyImageNet source images
  perlin_random     -- random Perlin patch (no CLIP), class from TinyImageNet attack_list
  perlin_noaccess   -- random patch + random CIFAR class (no surrogate data at all)
"""

import argparse
import os
import random

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from scipy.stats import beta

import forest
import UAP_attack_UBWC as uap
from relabel import build_text_class_mapping


def get_parser():
    parser = argparse.ArgumentParser(
        description="Cross-dataset UAP attack (TinyImageNet -> CIFAR-100 victim)"
    )
    parser.add_argument("--net", default="ResNet18", type=lambda s: [str(item) for item in s.split(",")])
    parser.add_argument("--dataset", default="CIFAR100", type=str)
    parser.add_argument("--recipe", default="gradient-matching", type=str)
    parser.add_argument("--threatmodel", default="single-class", type=str)
    parser.add_argument("--data_path", default="./data/data", type=str)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--exp_index", default=0, type=int)

    parser.add_argument("--tiny_train_path", default="./data/tiny/train/", type=str)
    parser.add_argument("--tiny_words_path", default="./data/tiny/words.txt", type=str)
    parser.add_argument("--tiny_wnids_path", default="./data/tiny/wnids.txt", type=str)
    parser.add_argument("--cifar_train_path", default="./data/cifar100/train/", type=str)
    parser.add_argument("--test_path", default="./data/cifar100/test/", type=str)
    parser.add_argument("--backdoored_test_path", default="./data/cifar100/backdoored_test/", type=str)
    parser.add_argument("--target_model_path", type=str, default="./data/cifar100/target_full/target_model.pth")
    parser.add_argument("--clip_cache_dir", default="./data/model/", type=str)
    parser.add_argument("--class_sim_threshold", type=float, default=0.7)

    parser.add_argument("--attack_method", default="perlin_uap_paste", type=str,
                        choices=["perlin_uap_paste", "perlin_random", "perlin_noaccess"])
    parser.add_argument("--patch_size", default=8, type=int)
    parser.add_argument("--uap_perlin_candidates", default=100, type=int,
                        help="Perlin candidates to screen (perlin_uap_paste only)")
    parser.add_argument("--class_imbalances", default="2,4", type=str)
    parser.add_argument("--list_sizes", default="1000,5000,10000", type=str)
    parser.add_argument("--num_trials", default=10, type=int)

    parser.add_argument("--use_tmp", action="store_true")
    parser.add_argument("--tmp_dir", default="/tmp/uap_cross", type=str)
    parser.add_argument("--mapping_only", action="store_true",
                        help="Print CLIP text class mapping and exit")
    parser.add_argument("--deterministic", action="store_true")
    return parser


# ---------------------------------------------------------------------------
# TinyImageNet helpers
# ---------------------------------------------------------------------------

def _cifar_class_names(cifar_train_path):
    return sorted(
        name for name in os.listdir(cifar_train_path)
        if os.path.isdir(os.path.join(cifar_train_path, name))
    )


def _tiny_image_dir(tiny_train_path, wnid):
    """Resolve the actual directory containing images for a wnid."""
    images_dir = os.path.join(tiny_train_path, wnid, "images")
    if os.path.isdir(images_dir):
        return images_dir
    return os.path.join(tiny_train_path, wnid)


def _list_tiny_images(tiny_train_path, wnid):
    """List image filenames (basenames) for one wnid."""
    return uap._list_image_files(_tiny_image_dir(tiny_train_path, wnid))


def _build_tiny_img_list(tiny_train_path, matched_wnids):
    """Build full (wnid, filename) list across all matched wnids."""
    all_img_list = []
    for wnid in matched_wnids:
        for fname in _list_tiny_images(tiny_train_path, wnid):
            all_img_list.append((wnid, fname))
    return all_img_list


def _majority_class(attack_list):
    counts = {}
    for class_name, _ in attack_list:
        counts[class_name] = counts.get(class_name, 0) + 1
    return max(counts, key=counts.get)


def _sample_attack_list_from_tiny(
    tiny_train_path, listing, class_imbalance, list_size, number_per_class
):
    """Beta-distribution class-imbalance sampling from TinyImageNet wnids."""
    if class_imbalance == 1:
        all_img_list = _build_tiny_img_list(tiny_train_path, listing)
        return random.sample(all_img_list, min(list_size, len(all_img_list)))

    old = 0
    attack_list = []
    classes = listing.copy()
    random.shuffle(classes)
    for j in range(1, len(classes) + 1):
        cdf_value = beta.cdf(j / len(classes), class_imbalance, class_imbalance)
        class_name = classes[j - 1]
        sample_list = _list_tiny_images(tiny_train_path, class_name)
        n = min(int((cdf_value - old) * list_size) + 1, len(sample_list), number_per_class)
        if n > 0 and sample_list:
            chosen = random.sample(sample_list, n)
            attack_list += [(class_name, fname) for fname in chosen]
        old = cdf_value
    return attack_list


# ---------------------------------------------------------------------------
# Cross-dataset CLIP-scored Perlin patch attack
# ---------------------------------------------------------------------------

def _load_tiny_source_images(attack_list, tiny_train_path, source_wnid):
    """Load TinyImageNet source images for CLIP scoring (kept at native 64x64)."""
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
    ])
    source_images = []
    for cls, fname in attack_list:
        if cls == source_wnid:
            img_path = os.path.join(
                _tiny_image_dir(tiny_train_path, cls), fname
            )
            img = to_tensor(Image.open(img_path).convert("RGB")).unsqueeze(0).cuda()
            source_images.append(img)
    return torch.cat(source_images, dim=0)


def attack_perlin_paste_cross(attack_list, args, tiny_train_path):
    """CLIP-scored Perlin patch using TinyImageNet source images.

    Same logic as _search_perlin_patch in UAP_attack_UBWC.py, but loads
    images from TinyImageNet (resized to 32x32) instead of CIFAR-100.
    Returns (source_wnid, trigger_PIL, mask_np).
    """
    import open_clip

    source_wnid = _majority_class(attack_list)
    print(source_wnid)

    cache_dir = getattr(args, "clip_cache_dir", "./data/model/")
    os.makedirs(cache_dir, exist_ok=True)
    for pretrained in ["datacomp_xl_s13b_b90k", "openai"]:
        try:
            try:
                base_model, _, _ = open_clip.create_model_and_transforms(
                    "ViT-B-16", pretrained=pretrained,
                    device="cuda:0", cache_dir=cache_dir,
                )
            except TypeError:
                base_model, _, _ = open_clip.create_model_and_transforms(
                    "ViT-B-16", pretrained=pretrained, device="cuda:0",
                )
            break
        except RuntimeError:
            base_model = None
    if base_model is None:
        raise RuntimeError("Failed to load CLIP model")
    clip_encode = base_model.encode_image
    for p in base_model.parameters():
        p.requires_grad = False
    base_model.eval()

    processor = transforms.Compose([
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
        transforms.Resize(224),
    ])
    to_tensor = transforms.Compose([transforms.ToTensor()])

    source_images = _load_tiny_source_images(
        attack_list, tiny_train_path, source_wnid
    )

    img_size = 32
    patch_size = getattr(args, "patch_size", 8)
    n_candidates = getattr(args, "uap_perlin_candidates", 100)
    position = "center"

    noises = [
        uap.generate_noise_image((3, patch_size, patch_size))
        for _ in range(n_candidates)
    ]

    r0, c0 = uap._patch_origin(img_size, patch_size, position)
    r1, c1 = r0 + patch_size, c0 + patch_size
    mask_np = np.zeros((img_size, img_size, 3), dtype=np.float32)
    mask_np[r0:r1, c0:c1, :] = 1.0
    mask_t = to_tensor(
        Image.fromarray((mask_np * 255).astype(np.uint8))
    ).unsqueeze(0).cuda()

    batch_size = min(32, len(source_images))
    max_distance = 0
    best_trigger_np = None
    best_mask_np = None

    for noise_img in noises:
        trigger_np, candidate_mask = uap._build_patch_trigger_and_mask(
            noise_img, patch_size, img_size, position=position
        )
        trigger_t = to_tensor(
            Image.fromarray(trigger_np.astype(np.uint8))
        ).unsqueeze(0).cuda()

        dis = 0.0
        with torch.no_grad():
            for start in range(0, len(source_images), batch_size):
                batch = source_images[start:start + batch_size]
                clean_feat = clip_encode(processor(batch))
                perturbed = uap._perturb_with_patch(
                    batch, trigger_t, mask_t, paste=True, alpha=0.2
                )
                pert_feat = clip_encode(processor(perturbed))
                dis += torch.norm(clean_feat - pert_feat, dim=-1).sum().item()

        if dis > max_distance:
            max_distance = dis
            best_trigger_np = trigger_np
            best_mask_np = candidate_mask

    trigger = Image.fromarray(best_trigger_np.astype(np.uint8))
    return source_wnid, trigger, best_mask_np


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_ubwc(trigger, source_cifar_class, args, mask):
    return uap.UWBC_test(trigger, source_cifar_class, args, mask=mask, paste=True)


def main():
    args = get_parser().parse_args()
    if args.deterministic:
        forest.utils.set_deterministic()

    cifar_classes = _cifar_class_names(args.cifar_train_path)

    if args.mapping_only:
        build_text_class_mapping(
            cifar_classes,
            args.tiny_words_path,
            args.tiny_wnids_path,
            threshold=args.class_sim_threshold,
            clip_cache_dir=args.clip_cache_dir,
        )
        return

    if args.attack_method != "perlin_noaccess":
        mapping, _, wnid_names = build_text_class_mapping(
            cifar_classes,
            args.tiny_words_path,
            args.tiny_wnids_path,
            threshold=args.class_sim_threshold,
            clip_cache_dir=args.clip_cache_dir,
        )
        if not mapping:
            raise RuntimeError(
                "No wnids matched at threshold {}. Lower --class_sim_threshold.".format(
                    args.class_sim_threshold
                )
            )
    else:
        mapping = {}

    if not os.path.isfile(args.target_model_path):
        raise FileNotFoundError(
            "Victim checkpoint not found: {}. Train with:\n"
            "  python3 train_target_model.py --target_train_path ./data/cifar100/train/".format(
                args.target_model_path
            )
        )

    setup = forest.utils.system_startup(args)
    model = forest.Victim(args, setup=setup)
    data = forest.Kettle(args, model.defs.batch_size, model.defs.augmentations, setup=setup)
    data_mean, data_std = data.trainset.data_mean, data.trainset.data_std

    uap.data_mean = data_mean
    uap.data_std = data_std

    args.classes = data.trainset.classes
    args.target_path = "./data/cifar100/target({})/".format(args.exp_index)

    if args.use_tmp:
        uap._setup_tmp_workspace(args)

    matched_wnids = sorted(mapping.keys())
    number_per_class = 500

    # -- perlin_noaccess: random class, random patch, no surrogate data -----
    if args.attack_method == "perlin_noaccess":
        results_all = 0
        for _ in range(args.num_trials):
            source_cifar = random.choice(cifar_classes)
            print(source_cifar)
            trigger, mask = uap.attack_perlin_noaccess(source_cifar, args)
            results_all += _run_ubwc(trigger, source_cifar, args, mask)
        print("{}/{} detections".format(results_all, args.num_trials))
        print("-------------Job finished.-------------------------")
        return

    # -- cross-dataset methods: perlin_uap_paste or perlin_random -----------
    class_imbalances = [
        int(x.strip()) for x in args.class_imbalances.split(",") if x.strip()
    ]
    list_sizes = [int(x.strip()) for x in args.list_sizes.split(",") if x.strip()]

    for class_imbalance in class_imbalances:
        for list_size in list_sizes:
            print("================= class_imbalance: {} | list_size: {} ===================".format(
                class_imbalance, list_size))

            results_all = 0
            for _ in range(args.num_trials):
                attack_list = _sample_attack_list_from_tiny(
                    args.tiny_train_path,
                    matched_wnids,
                    class_imbalance,
                    list_size,
                    number_per_class,
                )
                if not attack_list:
                    print("warning: empty attack_list, skipping trial")
                    continue

                if args.attack_method == "perlin_uap_paste":
                    source_wnid, trigger, mask = attack_perlin_paste_cross(
                        attack_list, args, args.tiny_train_path
                    )
                elif args.attack_method == "perlin_random":
                    source_wnid = _majority_class(attack_list)
                    print(source_wnid)
                    trigger, mask = uap.attack_perlin_noaccess(
                        mapping[source_wnid], args
                    )

                source_cifar = mapping[source_wnid]
                print("  {} ({}) -> {}".format(
                    source_wnid, wnid_names.get(source_wnid, "?"), source_cifar
                ))
                results_all += _run_ubwc(trigger, source_cifar, args, mask)

            print(results_all)

    print("-------------Job finished.-------------------------")


if __name__ == "__main__":
    main()
