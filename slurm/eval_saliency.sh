#!/bin/bash
#SBATCH --job-name=sal-eval
#SBATCH --account=ACCOUNT
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=job_%x_%A_%a.out
#SBATCH --gpus-per-node=h100:1
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --array=0-5

# Saliency-baseline ADE20K mIoU eval (1 full-scene glimpse + T saliency glimpses).
#
# Usage:
#   export COMET_API_KEY=...   # sbatch forwards the submitting shell's env
#   sbatch slurm/eval_saliency.sh                  # all runs in --array (deepgaze + gbvs)
#   sbatch --array=6-8 slurm/eval_saliency.sh      # pick runs by index from RUNS
#   sbatch --array=0 slurm/eval_saliency.sh deepgaze blobcont   # explicit single run
#
# METHOD: itti | gbvs | deepgaze  (cache/saliency/ade20k_validation/<METHOD>)
# MODE:
#   fixed05      fixed_fovea, scale 0.5
#   blobcont     blob_continuous, scale in [0.25, 1.0]
#   blob025-05   blob, quantized to {0.25, 0.5}
# T: saliency glimpses after t0 (default 4)

set -euo pipefail

RUNS=(
  "deepgaze fixed05"     # 0
  "deepgaze blobcont"    # 1
  "deepgaze blob025-05"  # 2
  "gbvs fixed05"         # 3
  "gbvs blobcont"        # 4
  "gbvs blob025-05"      # 5
  "itti fixed05"         # 6
  "itti blobcont"        # 7
  "itti blob025-05"      # 8
)

if [[ $# -ge 2 ]]; then
  METHOD=$1
  MODE=$2
else
  TASK_ID=${SLURM_ARRAY_TASK_ID:?no METHOD MODE given and not running as an array task}
  read -r METHOD MODE <<< "${RUNS[$TASK_ID]:?no RUNS entry for array index $TASK_ID}"
fi
T=${3:-4}

case "$MODE" in
  fixed05)    MODE_ARGS=(--selection-mode fixed_fovea --scales 0.5) ;;
  blobcont)   MODE_ARGS=(--selection-mode blob_continuous --min-scale 0.25 --max-scale 1.0) ;;
  blob025-05) MODE_ARGS=(--selection-mode blob --scales 0.25,0.5 --blob-margin 1.25) ;;
  *) echo "unknown MODE: $MODE" >&2; exit 1 ;;
esac

TAG="${METHOD}-${MODE}-t${T}"

module load apptainer/1.4.5

export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export CANVIT_CHECKPOINT="canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02"
: "${COMET_API_KEY:?export COMET_API_KEY before sbatch}"

mkdir -p "$SLURM_TMPDIR/dataset"
unzip -q /project/6061451/datasets/ade20k/ADEChallengeData2016.zip -d "$SLURM_TMPDIR/dataset"

apptainer exec --nv --writable-tmpfs \
  --bind "$SLURM_TMPDIR/dataset:/data" \
  --bind "/project/6061451/$USER/CanViT-rl:/workspace" \
  --bind "$SCRATCH/hf_cache:$SCRATCH/hf_cache" \
  --bind /etc/pki/tls/cert.pem:/etc/pki/tls/certs/ca-bundle.crt \
  --env HF_HOME="$SCRATCH/hf_cache" \
  --env SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt \
  --env HF_HUB_OFFLINE=1 \
  --env CANVIT_CHECKPOINT="$CANVIT_CHECKPOINT" \
  --env COMET_API_KEY="$COMET_API_KEY" \
  --env PYTHONPATH=/workspace \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env PYTHONPYCACHEPREFIX=/tmp/pycache \
  "$SCRATCH/canvit_rl.sif" \
  python -u /workspace/scripts/saliency/eval_saliency_baseline_miou.py \
    --method "$METHOD" \
    "${MODE_ARGS[@]}" \
    --t "$T" \
    --dataset /data/ADEChallengeData2016 \
    --split validation \
    --batch-size 8 \
    --num-workers 8 \
    --visualize-samples 4 \
    --visualization-dir "results/saliency_visualizations/${TAG}" \
    --output "results/json/saliency_${TAG}.json" \
    --comet \
    --comet-project eval-saliency \
    --experiment-name "$TAG"
