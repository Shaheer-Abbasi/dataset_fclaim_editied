import argparse
import os
import shutil


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def is_image_file(path):
    return os.path.splitext(path)[1].lower() in IMG_EXTS


def safe_image_loader(path):
    from PIL import Image
    try:
        if not is_image_file(path):
            return None

        with Image.open(path) as img:
            return img.convert("RGB")

    except Exception:
        return None


def load_synset_file(txt_path):
    synset_dict = {}

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue

            synset_id, description = parts
            synset_dict[synset_id] = description

    return synset_dict


def load_tiny_wnid_names(words_path, wnids_path):
    words = load_synset_file(words_path)
    with open(wnids_path, "r", encoding="utf-8") as f:
        wnids = [line.strip() for line in f if line.strip()]
    return {
        wnid: words.get(wnid, wnid).split(",")[0].strip()
        for wnid in wnids
    }


def _format_class_prompt(name):
    return "a photo of {}".format(name.replace("_", " "))


def build_text_class_mapping(
    cifar_classes,
    words_path,
    wnids_path,
    threshold=0.7,
    clip_cache_dir="./data/model/",
    device=None,
    verbose=True,
):
    """Map TinyImageNet wnids to CIFAR-100 classes via CLIP text cosine similarity."""
    import open_clip
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    wnid_names = load_tiny_wnid_names(words_path, wnids_path)
    wnids = list(wnid_names.keys())

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16",
        pretrained="datacomp_xl_s13b_b90k",
        device=device,
        cache_dir=clip_cache_dir,
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-16")

    with torch.no_grad():
        tiny_texts = [_format_class_prompt(wnid_names[wnid]) for wnid in wnids]
        cifar_texts = [_format_class_prompt(name) for name in cifar_classes]

        tiny_tokens = tokenizer(tiny_texts).to(device)
        cifar_tokens = tokenizer(cifar_texts).to(device)

        tiny_features = model.encode_text(tiny_tokens)
        cifar_features = model.encode_text(cifar_tokens)

        tiny_features = tiny_features / tiny_features.norm(dim=-1, keepdim=True)
        cifar_features = cifar_features / cifar_features.norm(dim=-1, keepdim=True)

        sim_matrix = (tiny_features @ cifar_features.T).cpu()

    mapping = {}
    scores = {}
    for idx, wnid in enumerate(wnids):
        best_j = int(sim_matrix[idx].argmax())
        best_score = float(sim_matrix[idx, best_j])
        scores[wnid] = best_score
        if best_score >= threshold:
            mapping[wnid] = cifar_classes[best_j]

    if verbose:
        matched_cifar = set(mapping.values())
        print("Class mapping: {}/{} Tiny wnids matched (threshold={})".format(
            len(mapping), len(wnids), threshold))
        print("CIFAR-100 classes covered: {}/{}".format(
            len(matched_cifar), len(cifar_classes)))
        for wnid in sorted(mapping.keys()):
            print("  {} ({}) -> {} (sim={:.3f})".format(
                wnid, wnid_names[wnid], mapping[wnid], scores[wnid]))

    return mapping, scores, wnid_names


def run_image_relabel(args):
    import open_clip
    import torch
    from tqdm import tqdm

    label_text_to_word = load_tiny_wnid_names(args.tiny_words_path, args.tiny_wnids_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16",
        pretrained="datacomp_xl_s13b_b90k",
        device=device,
        cache_dir=args.clip_cache_dir,
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-16")

    if os.path.exists(args.output_path):
        shutil.rmtree(args.output_path)
    os.makedirs(args.output_path, exist_ok=True)

    text_dir = {}
    for wnid in os.listdir(args.tiny_train_path):
        class_dir = os.path.join(args.tiny_train_path, wnid)
        if not os.path.isdir(class_dir):
            continue
        if wnid not in label_text_to_word:
            continue
        text = [_format_class_prompt(label_text_to_word[wnid])]
        text_tokens = tokenizer(text).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_dir[wnid] = text_features
        os.makedirs(os.path.join(args.output_path, wnid), exist_ok=True)

    for class_name in os.listdir(args.cifar_train_path):
        class_dir = os.path.join(args.cifar_train_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        print(class_name)
        for fname in tqdm(os.listdir(class_dir)):
            with torch.no_grad():
                img = safe_image_loader(os.path.join(class_dir, fname))
                if img is None:
                    continue
                image = preprocess(img).unsqueeze(0).to(device)
                image_features = model.encode_image(image)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                for wnid, text_features in text_dir.items():
                    similarity = (image_features @ text_features.T).squeeze()
                    if float(similarity) > args.image_sim_threshold:
                        shutil.copy(
                            os.path.join(class_dir, fname),
                            os.path.join(args.output_path, wnid, fname),
                        )

    total = 0
    for wnid in text_dir:
        count = len(os.listdir(os.path.join(args.output_path, wnid)))
        print("{}: {}".format(wnid, count))
        total += count
    print("total copied:", total)


def get_parser():
    parser = argparse.ArgumentParser(description="TinyImageNet <-> CIFAR-100 CLIP mapping utilities")
    parser.add_argument("--mode", choices=["text_mapping", "image_relabel"], default="text_mapping")
    parser.add_argument("--cifar_train_path", default="./data/cifar100/train/")
    parser.add_argument("--tiny_train_path", default="./data/tiny/train/")
    parser.add_argument("--tiny_words_path", default="./data/tiny/words.txt")
    parser.add_argument("--tiny_wnids_path", default="./data/tiny/wnids.txt")
    parser.add_argument("--clip_cache_dir", default="./data/model/")
    parser.add_argument("--class_sim_threshold", type=float, default=0.7)
    parser.add_argument("--image_sim_threshold", type=float, default=0.25)
    parser.add_argument("--output_path", default="./data/cifar100/cifar100_tiny/")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    if args.mode == "text_mapping":
        cifar_classes = sorted(
            name for name in os.listdir(args.cifar_train_path)
            if os.path.isdir(os.path.join(args.cifar_train_path, name))
        )
        build_text_class_mapping(
            cifar_classes,
            args.tiny_words_path,
            args.tiny_wnids_path,
            threshold=args.class_sim_threshold,
            clip_cache_dir=args.clip_cache_dir,
        )
    else:
        run_image_relabel(args)
