#!/bin/bash
#SBATCH --job-name=Train-SAC
#SBATCH --account=ACCOUNT
#SBATCH --time=00:30:00                 # Request 4 hours (adjust as needed)
#SBATCH --cpus-per-task=8               # CPU cores for data loading
#SBATCH --mem=64G                       # System RAM memory
#SBATCH --output=job_%j.out             # Text log file named after the Job ID
#SBATCH --gpus-per-node=h100:1
#SBATCH --partition=gpubase_bygpu_b1

module load apptainer/1.4.5

export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1

export CANVIT_CHECKPOINT="canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02"
export COMET_API_KEY=COMET_API_KEY

mkdir -p $SLURM_TMPDIR/dataset
unzip -q /project/6061451/datasets/ade20k/ADEChallengeData2016.zip -d $SLURM_TMPDIR/dataset


apptainer exec --nv \
    --bind $SLURM_TMPDIR/dataset:/data \
    --bind /project/6061451/$USER/CanViT-rl:/workspace \
    --bind /etc/pki/tls/cert.pem:/etc/pki/tls/certs/ca-bundle.crt \
    --env HF_HOME=$SCRATCH/hf_cache \
    --env SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt \
    --env HF_HUB_OFFLINE=1 \
    --env PYTHONPATH=/workspace \
    --env PYTORCH_ALLOC_CON=expandable_segments:True \
    $SCRATCH/canvit_rl.sif \
    python -u /workspace/scripts/train_canvas_sac.py \
    --num-workers 8 \
    --dataset synthetic_segmentation \
    --dataset-format synthetic \
    --split training \
    --eval-split validation \
    --batches 10000 \
    --batch-size 4 \
    --max-samples 7 \
    --t 1 \
    --eval-images 7 \
    --eval-batch-size 1 \
    --reward-map-images 7 \
    --reward-map-interval 1000 \
    --replay-batch-size 16 \
    --learning-starts 512 \
    --tau 0.0035 \
    --init-alpha 0.015 \
    --alpha-lr 0.001 \
    --actor-lr 0.0013 \
    --critic-lr 0.0006 \
    --buffer-size 10000 \
    --comet-log-interval 1000 \
    --eval-interval 1000 \
    --experiment-name synthetic-im7-t1-10_000-critlr_3 \
    --checkpoint-dir checkpoints/canvas_sac/synthetic-im7-t1-10_000-critlr_3
