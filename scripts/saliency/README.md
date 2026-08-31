# ADE20K Saliency Baselines

This folder contains the saliency-map pipeline for heuristic CanViT viewpoint
baselines on ADE20K mIoU.

The pipeline has three stages:

1. Produce saliency maps.
2. Convert maps into the repo's `.pt` cache format.
3. Evaluate saliency-guided `Viewpoint`s with mIoU and optional multi-sample
   glimpse visualizations.

## Files

- `precompute_saliency_maps.py`: creates `.pt` saliency caches from native
  Python heuristics or external `.mat`/`.npy`/image maps.
- `eval_saliency_baseline_miou.py`: runs full-scene `t0`, then saliency-guided
  crops for `t1..tN`, and reports mIoU.
- `matlab/run_ade20k_saliency.m`: runs MATLAB saliency toolboxes over ADE20K
  images and saves `.mat` saliency maps plus preview PNGs.

## Native Python Maps

These do not require MATLAB:

```bash
uv run python scripts/saliency/precompute_saliency_maps.py \
  --method center_surround \
  --dataset datasets/ADE20k \
  --split validation
```

Other native methods:

```bash
--method edge
--method spectral_residual
```

Native outputs are saved under:

```text
cache/saliency/ade20k_validation/<method>/
```

## MATLAB Itti Setup

`--method itti` is reserved for Dirk Walther's SaliencyToolbox implementation
of the Itti/Koch/Walther saliency pipeline.

Clone SaliencyToolbox as a sibling of this repo:

```bash
cd <workspace-parent>
git clone https://github.com/DirkBWalther/SaliencyToolbox.git
```

Run MATLAB from the repo root:

```bash
cd <repo-root>

matlab -batch "addpath('scripts/saliency/matlab'); run_ade20k_saliency('method','itti','toolbox_root','../SaliencyToolbox')"
```

If `matlab` is not on your shell `PATH`, use the full macOS app path, adjusting
the version name and toolbox path:

```bash
/Applications/<MATLAB_VERSION>.app/bin/matlab -batch "addpath('scripts/saliency/matlab'); run_ade20k_saliency('method','itti','toolbox_root','../SaliencyToolbox')"
```

MATLAB writes:

```text
results/matlab_itti_maps/
  ADE_val_00000001.mat        # numeric salmap for conversion/eval
  previews/
    ADE_val_00000001.png      # 512x512 human preview
```

The `.mat` files keep the raw toolbox saliency values. Preview PNGs are only
for inspection.

## MATLAB GBVS Setup

`--method gbvs` can be exported with the MATLAB GBVS toolbox through
`matlab/run_ade20k_saliency.m`.

Download GBVS as a zip and place it as a sibling of this repo:

- Original GBVS page: `http://www.klab.caltech.edu/~harel/share/gbvs.php`
- GitHub mirror: `https://github.com/Pinoshino/gbvs`
- Direct GitHub zip: `https://github.com/Pinoshino/gbvs/archive/refs/heads/master.zip`

```bash
cd <workspace-parent>
# Unzip the download so this folder exists:
# <workspace-parent>/gbvs/gbvs.m
```

The expected local layout is:

```text
<workspace-parent>/
  CanViT-rl/
  gbvs/
    gbvs.m
    gbvs_install.m
    compile/
    saltoolbox/
    util/
```

Required MATLAB products:

- MATLAB
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox, used by GBVS helpers such as
  `normpdf`

On macOS, GBVS also needs a supported Xcode compiler to build MEX helpers such
as `mySubsample`. After installing/configuring Xcode, run MATLAB's compiler
setup:

```bash
/Applications/<MATLAB_VERSION>.app/bin/matlab -batch "mex -setup C++"
```

Install and compile GBVS from the GBVS root:

```bash
/Applications/<MATLAB_VERSION>.app/bin/matlab -batch "restoredefaultpath; cd('<workspace-parent>/gbvs'); addpath(genpath(pwd)); addpath(fullfile(pwd,'compile')); gbvs_install; gbvs_compile; which mySubsample -all; which gbvs -all"
```

`which mySubsample -all` should print a compiled MEX file under
`<workspace-parent>/gbvs/saltoolbox/`. On Apple silicon this is typically a
`.mexmaca64` file.

Run MATLAB from the repo root:

```bash
cd <repo-root>

matlab -batch "addpath('scripts/saliency/matlab'); run_ade20k_saliency('method','gbvs','toolbox_root','../gbvs')"
```

Because the runner defaults GBVS to the sibling `<workspace-parent>/gbvs`
folder, this shorter command is equivalent when GBVS is installed there:

```bash
matlab -batch "addpath('scripts/saliency/matlab'); run_ade20k_saliency('method','gbvs')"
```

On macOS, replace `<MATLAB_VERSION>` with the installed MATLAB app name.

```bash
/Applications/<MATLAB_VERSION>.app/bin/matlab -batch "restoredefaultpath; cd('<repo-root>'); addpath('scripts/saliency/matlab'); run_ade20k_saliency('method','gbvs')"
```

If stale MATLAB path warnings mention an old GBVS folder, reset the saved path:

```bash
/Applications/<MATLAB_VERSION>.app/bin/matlab -batch "restoredefaultpath; savepath"
```

MATLAB writes:

```text
results/matlab_gbvs_maps/
  ADE_val_00000001.mat
  previews/
    ADE_val_00000001.png
```

Convert those maps into the Python cache:

```bash
uv run python scripts/saliency/precompute_saliency_maps.py \
  --method gbvs \
  --external-map-dir results/matlab_gbvs_maps \
  --dataset datasets/ADE20k \
  --split validation
```

## DeepGaze Setup

DeepGaze maps are exported with the official `deepgaze_pytorch` implementation.
The saliency script uses DeepGaze IIE by default because it is a spatial
saliency/fixation-density model that does not require a previous scanpath.

Required Python packages:

- `torch` and `torchvision`, already part of the main project environment
- `numpy`, `Pillow`, and `tqdm`, already part of the saliency scripts'
  runtime path
- `deepgaze_pytorch`, installed from the official DeepGaze repository
- `clip`, required because the current DeepGaze package imports its MSDB/CLIP
  modules at package import time
- `einops`, required because the current DeepGaze package imports its
  MSDB/DINO modules at package import time

Install the DeepGaze-specific packages into the project environment:

```bash
uv pip install einops \
  "git+https://github.com/openai/CLIP.git" \
  "git+https://github.com/matthias-k/DeepGaze.git"
```

Optional one-image smoke test:

```bash
uv run python scripts/saliency/export_deepgaze_maps.py \
  --dataset datasets/ADE20k \
  --split validation \
  --output-dir results/deepgaze_maps_smoke \
  --max-images 1 \
  --device cpu \
  --overwrite
```

Export ADE20K validation maps:

```bash
uv run python scripts/saliency/export_deepgaze_maps.py \
  --dataset datasets/ADE20k \
  --split validation \
  --output-dir results/deepgaze_maps
```

The first run downloads DeepGaze and torchvision backbone weights through
PyTorch's model cache. This includes the DeepGaze IIE checkpoint and backbone
weights such as:

```text
deepgaze2e.pth
resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar
efficientnet-b5-b6417697.pth
densenet201-c1103571.pth
```

If a network blocks `download.pytorch.org`, download the missing checkpoint
manually into the torch cache path printed by the error, then rerun the export
command. Use `--device cpu`, `--device cuda`, or `--device mps` to force a
backend; the default is `auto`.

For example, if the missing file is `densenet201-c1103571.pth`, place it in the
torch checkpoint cache:

```bash
mkdir -p ~/.cache/torch/hub/checkpoints
curl -L \
  -o ~/.cache/torch/hub/checkpoints/densenet201-c1103571.pth \
  https://download.pytorch.org/models/densenet201-c1103571.pth
```

If DNS for `download.pytorch.org` fails in one terminal/session, try the same
command from a browser download or another network, then move the file to the
cache path above.

Recommended output layout:

```text
results/deepgaze_maps/
  ADE_val_00000001.npy
  ADE_val_00000002.npy
  previews/
    ADE_val_00000001.png
```

The exporter writes `.npy` numeric maps and preview PNGs. It uses a uniform
log-density center bias by default so the cache is driven by DeepGaze image
features rather than ADE-specific fixation data.

After exporting DeepGaze maps:

```bash
uv run python scripts/saliency/precompute_saliency_maps.py \
  --method deepgaze \
  --external-map-dir results/deepgaze_maps \
  --dataset datasets/ADE20k \
  --split validation
```

That conversion writes the standard cache used by the mIoU evaluator:

```text
cache/saliency/ade20k_validation/deepgaze/
```

## Convert External Maps

Convert MATLAB `.mat` maps into `.pt` cache files:

```bash
uv run python scripts/saliency/precompute_saliency_maps.py \
  --method itti \
  --external-map-dir results/matlab_itti_maps \
  --dataset datasets/ADE20k \
  --split validation
```

The converter also accepts `.pt`, `.npy`, `.png`, `.jpg`, `.tif`, and `.tiff`
external maps. External map filenames must match ADE image stems, for example:

```text
datasets/ADE20k/images/validation/ADE_val_00000001.jpg
results/matlab_itti_maps/ADE_val_00000001.mat
```

For other external methods, save one map per ADE image stem, then convert with:

```bash
uv run python scripts/saliency/precompute_saliency_maps.py \
  --method gbvs \
  --external-map-dir results/gbvs_maps \
  --dataset datasets/ADE20k \
  --split validation
```

Supported external method labels:

```text
deepgaze
gbvs
itti
```

## Evaluate mIoU

Evaluate cached maps:

```bash
uv run python scripts/saliency/eval_saliency_baseline_miou.py \
  --method itti \
  --t 5 \
  --scales 0.25 \
  --dataset datasets/ADE20k
```

The policy convention is:

```text
t0: full-scene Viewpoint
t1: most salient crop
t2: next most salient crop after NMS suppression
...
tN: next selected crop
```

With multiple candidate scales, the policy picks the highest average-saliency
window across the provided scales:

```bash
--scales 0.15,0.25,0.35
```

### Blob-Sized Zoom

Use connected saliency blobs to choose the crop center and zoom. The blob
bounding box is expanded by `--blob-margin`, then quantized to the smallest
allowed scale from `--scales` that covers it:

```bash
uv run python scripts/saliency/eval_saliency_baseline_miou.py \
  --method itti \
  --selection-mode blob \
  --t 5 \
  --scales 0.25,0.5 \
  --blob-threshold-quantile 0.85 \
  --blob-margin 1.25 \
  --dataset datasets/ADE20k \
  --visualize-samples 4
```

For example, a compact blob will use `0.25` if it fits after the margin; a
larger blob will use `0.5`. If a blob exceeds all allowed scales, the largest
provided scale is used.

## Visualize First Samples

Save one multi-row figure for the first few images:

```bash
uv run python scripts/saliency/eval_saliency_baseline_miou.py \
  --method itti \
  --t 5 \
  --scales 0.25 \
  --dataset datasets/ADE20k \
  --visualize-samples 4
```

Outputs:

```text
results/saliency_visualizations/
  itti_saliency_timesteps.png
```

The figure matches the IN21k SAC glimpse layout: each row is one sample, the
first column overlays all selected boxes, and the remaining columns show one
timestep each:

```text
all viewpoints | t0 full scene | t1 crop | t2 crop | ... | t5 crop
```

Post-`t0` columns lightly blend the saliency heatmap over the image and draw the
current timestep's selected box.

## Log To Comet

Comet logging is opt-in:

```bash
uv run python scripts/saliency/eval_saliency_baseline_miou.py \
  --method itti \
  --selection-mode blob \
  --t 5 \
  --scales 0.25,0.5 \
  --dataset datasets/ADE20k \
  --visualize-samples 4 \
  --comet \
  --comet-project canvit-rl \
  --experiment-name itti-blob-saliency
```

The script logs:

- final scalar metrics such as `miou/t0`, `miou/t5`, `miou/final`, `scale/t0`,
  and `scale/t5`
- the saved JSON result payload as a Comet asset
- the combined visualization figure when `--visualize-samples` is positive

## Cache Formats

- `.mat`: accurate MATLAB-side saliency values, usually variable name `salmap`.
- `.pt`: Python-side cache used by mIoU eval.
- result `.json`: mIoU, mean scale, and eval metadata saved by
  `eval_saliency_baseline_miou.py`.
- preview `.png`: human inspection only.
