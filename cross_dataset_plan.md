# Cross-Dataset False-Claim Attack: TinyImageNet to CIFAR-100

## Why this matters

Right now the attacker and victim both use CIFAR-100 data, which is an unrealistically easy setting. In a real false-claim scenario, the attacker would not have access to the victim's exact training distribution. By designing the patch on TinyImageNet and testing it against a CIFAR-100 victim, we demonstrate the attack transfers across datasets -- a much stronger claim.

## High-level data flow

1. Load TinyImageNet images (64x64)
2. Use CLIP text encoding to map TinyImageNet classes to CIFAR-100 classes (create a matched subset with 10-50% class overlap)
3. Design Perlin patch using TinyImageNet source images (resize to 32x32) via `_search_perlin_patch`
4. Apply the resulting patch to CIFAR-100 test images (32x32)
5. Run UBWC on the CIFAR-100 victim model (ResNet18)
6. Measure false positive rate

## Step 1: CLIP text-based class mapping

TinyImageNet has 200 classes identified by WordNet IDs (e.g., `n02099601` = "golden retriever"). CIFAR-100 has 100 human-readable classes (e.g., "beaver", "oak_tree"). Many TinyImageNet classes are finer-grained versions of CIFAR-100 classes.

**Implementation:** Create a function `build_class_mapping()` that:

1. Loads TinyImageNet's `words.txt` to get `{wnid: human_name}` (e.g., `n02099601 -> "golden retriever"`)
2. Gets CIFAR-100 class names from `args.classes` (already available)
3. Uses CLIP's text encoder to embed both sets of class names: `open_clip.get_tokenizer('ViT-B-16')` + `model.encode_text()`
4. Computes cosine similarity matrix (200 x 100)
5. For each TinyImageNet class, finds the best-matching CIFAR-100 class and the similarity score
6. Filters by a configurable threshold to produce a mapping: `{wnid: cifar100_class_name}`
7. Prints the mapping and overlap stats so we can inspect it

This gives us, for example: `n02099601 ("golden retriever") -> "wolf" (similarity: 0.82)`. We can tune the threshold to get 10-50% overlap (10-50 CIFAR-100 classes covered).

**Location:** New function in `UAP_attack_UBWC.py`, near the top with other utility functions (around line 157).

## Step 2: New CLI arguments

Add to the argument parser (around line 620):

- `--surrogate_img_path` (str, default `None`): Path to TinyImageNet exported train images (e.g., `./data/tiny/train/`). When set, enables cross-dataset mode.
- `--tiny_words_path` (str, default `./tiny-imagenet-200/words.txt`): Path to TinyImageNet `words.txt` for wnid-to-name mapping.
- `--class_sim_threshold` (float, default `0.7`): Minimum CLIP text cosine similarity for a TinyImageNet class to be considered a match for a CIFAR-100 class.
- `--surrogate_img_size` (int, default `64`): Native resolution of surrogate images (for TinyImageNet, 64).

## Step 3: Modify source image loading for cross-dataset

**Resolution handling:** Resize TinyImageNet images from 64x64 to 32x32 when loading them as source images. This is the simplest approach because:
- The Perlin patch is generated at 8x8 and placed on a 32x32 canvas -- no changes needed to `_search_perlin_patch` or `_build_patch_trigger_and_mask`
- CLIP resizes everything to 224x224 anyway, so 32 vs 64 input makes negligible difference for scoring
- The resulting trigger is immediately usable on CIFAR-100 test images (also 32x32)
- We can always revisit the "design at 64, downscale later" approach if results look bad

**Changes to `_load_surrogate_and_source_images` (line 158):**
- Accept an optional `surrogate_img_path` parameter
- When cross-dataset mode is active, load images from `surrogate_img_path` instead of `args.img_path`
- Add a `Resize(32)` transform before `ToTensor()` when loading 64x64 images
- The source class validation uses the surrogate path's class folders instead of CIFAR-100's

**Changes to `_attack_class_names` (line 148):**
- When cross-dataset, list TinyImageNet wnid folders under `surrogate_img_path` instead of CIFAR-100 human-name folders

## Step 4: Modify main loop for cross-dataset attack_list

In the `__main__` block (lines 655-710), when `args.surrogate_img_path` is set:

1. Run `build_class_mapping()` once at the top to get the `{wnid: cifar100_class}` mapping
2. Build `args.all_img_list` from TinyImageNet images in the matched wnid folders (instead of CIFAR-100 train images)
3. Use `listing` = matched TinyImageNet class names (wnids)
4. After `_search_perlin_patch` returns a TinyImageNet `source_class` (a wnid), map it back to the corresponding CIFAR-100 class name for `UBWC_test`

The `UBWC_test` and `create_backdoor_testset` functions remain unchanged -- they still operate on CIFAR-100 test images and the CIFAR-100 victim model.

## Step 5: Attack method choices

Keep `"perlin_uap_paste"` and auto-detect cross-dataset mode from `--surrogate_img_path`. Same attack logic, different source data.

### New method: `perlin_random`

Add a new `--attack_method` choice `"perlin_random"` that **skips CLIP scoring entirely**. Instead of generating 100 Perlin candidates and picking the one with maximum CLIP feature displacement, it just generates a single random Perlin patch and uses it directly.

This is an ablation to test whether the CLIP-based optimization actually matters, or if any random Perlin texture is sufficient to fool UBWC. If random works just as well, the attack is even simpler, cheaper, and harder to defend against.

**Implementation:** New function `attack_perlin_random(attack_list, args)` that:
1. Picks the source class from the attack list (same as before)
2. Generates one random Perlin noise image at `patch_size` resolution
3. Builds trigger and mask at center position (same as `perlin_uap_paste`)
4. Returns `(source_class, trigger, mask)` -- no CLIP model loaded at all

This is ~10 lines of code since it reuses `_build_patch_trigger_and_mask` and `generate_noise_image`.

## Step 6: Train victim model on full CIFAR-100 (50k images)

Currently the victim ResNet18 is trained on a 50% split (`target(0)/` = 25,000 images). For the cross-dataset experiments, train on the full 50,000-image CIFAR-100 training set. This produces a stronger victim model, making the attack more convincing.

**How:** Use the existing `train_target_model.py` script but point it at the full training set instead of a target split. The resulting model checkpoint will be used for all cross-dataset experiments.

We need to train this model before running the cross-dataset attack grid.

## What does NOT change

- `create_backdoor_testset` -- still copies and patches CIFAR-100 test images
- `UWBC_test` -- still loads the CIFAR-100 victim ResNet18 and runs paired t-test
- `_perturb_with_patch`, `_build_patch_trigger_and_mask`, `_patch_origin` -- all unchanged
- Perlin noise generation -- same parameters

## Patch size experiments

Lab mate requested: try both 8x8 and 16x16 patches.

### Literature reference

"Meta Adversarial Training against Universal Patches" (ICLR) uses **24x24 patches on 64x64 TinyImageNet images** (~14% coverage). This is the standard for TinyImageNet adversarial patch research.

### Coverage comparison

- 8x8 on 32x32 (current CIFAR-100 setup): **6.25%**
- 8x8 on 64x64 (TinyImageNet, no resize): **1.56%** -- likely too weak
- 16x16 on 64x64 (TinyImageNet, no resize): **6.25%** -- matches current CIFAR-100 setup
- 16x16 on 32x32 (TinyImageNet, resized): **25%** -- aggressive but effective
- 24x24 on 64x64 (literature standard): **14%**

### Plan

Run two configurations with `--patch_size`:
1. `--patch_size 8` (6.25% coverage at 32x32)
2. `--patch_size 16` (25% coverage at 32x32)

The `--patch_size` argument already exists in the CLI, so no new code is needed for this -- just run the experiment grid twice.

### Resize decision (pending lab mate response)

If we resize TinyImageNet to 32x32:
- 8x8 = 6.25% (consistent with current CIFAR-100 experiments)
- 16x16 = 25% (strong attack, may be too aggressive)

If we do NOT resize (keep 64x64):
- 8x8 = 1.56% (probably too small to be effective)
- 16x16 = 6.25% (matches current CIFAR-100 8x8 proportionally)

Leaning toward resize-to-32 for simplicity, but waiting for lab mate's input.

## Experiment matrix

We now have 3 variables to sweep:
- **Attack method:** `perlin_uap_paste` (CLIP-optimized) vs `perlin_random` (no CLIP)
- **Patch size:** 8x8 vs 16x16
- **Dataset mode:** cross-dataset (TinyImageNet surrogate) -- all runs use this

That gives us 4 experiment runs (2 methods x 2 patch sizes), each sweeping the usual class_imbalance x list_size grid.

## Example commands

```bash
# CLIP-optimized, patch size 8x8
python3 UAP_attack_UBWC.py \
  --net ResNet18 --dataset CIFAR100 \
  --attack_method perlin_uap_paste \
  --surrogate_img_path ./data/tiny/train/ \
  --tiny_words_path ./tiny-imagenet-200/words.txt \
  --class_sim_threshold 0.7 \
  --patch_size 8 \
  --class_imbalances 2,3,4 \
  --list_sizes 1000,2000,5000,10000 \
  --num_trials 20 \
  2>&1 | tee logs/cross_dataset_paste_8x8.log

# Patch size 16x16
python3 UAP_attack_UBWC.py \
  --net ResNet18 --dataset CIFAR100 \
  --attack_method perlin_uap_paste \
  --surrogate_img_path ./data/tiny/train/ \
  --tiny_words_path ./tiny-imagenet-200/words.txt \
  --class_sim_threshold 0.7 \
  --patch_size 16 \
  --class_imbalances 2,3,4 \
  --list_sizes 1000,2000,5000,10000 \
  --num_trials 20 \
  2>&1 | tee logs/cross_dataset_paste_16x16.log

# Random (no CLIP), patch size 8x8
python3 UAP_attack_UBWC.py \
  --net ResNet18 --dataset CIFAR100 \
  --attack_method perlin_random \
  --surrogate_img_path ./data/tiny/train/ \
  --tiny_words_path ./tiny-imagenet-200/words.txt \
  --class_sim_threshold 0.7 \
  --patch_size 8 \
  --class_imbalances 2,3,4 \
  --list_sizes 1000,2000,5000,10000 \
  --num_trials 20 \
  2>&1 | tee logs/cross_dataset_random_8x8.log

# Random (no CLIP), patch size 16x16
python3 UAP_attack_UBWC.py \
  --net ResNet18 --dataset CIFAR100 \
  --attack_method perlin_random \
  --surrogate_img_path ./data/tiny/train/ \
  --tiny_words_path ./tiny-imagenet-200/words.txt \
  --class_sim_threshold 0.7 \
  --patch_size 16 \
  --class_imbalances 2,3,4 \
  --list_sizes 1000,2000,5000,10000 \
  --num_trials 20 \
  2>&1 | tee logs/cross_dataset_random_16x16.log
```
