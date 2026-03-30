#!/usr/bin/env bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=7
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export NCCL_P2P_DISABLE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

GPU_NUM=1
output_dir="../output/"
weight_path="/20TB/yanchengzhi/jinjiandong/Tracking/output/LORAT_LUART-dinov2-2026.01.06-09.10.15-355200/checkpoint/epoch_09/model.bin"
timestamp=$(date +"%Y.%m.%d-%H.%M.%S")

mkdir -p "$output_dir"

#dry-run
#python ../main.py GOLA dinov2 --dry_run --distributed_nproc_per_node "${GPU_NUM}" --distributed_do_spawn_workers --disable_wandb --weight_path $weight_path

# GOLA-B
python main.py --method_name LORAT_LUART --mixin evaluation --config_name dinov2 --distributed_nproc_per_node "${GPU_NUM}" --distributed_do_spawn_workers --disable_wandb --weight_path $weight_path --output_dir="$output_dir" |& tee -a "$output_dir/train_stdout-$timestamp.log"

# GOLA-L
# python ../main.py GOLA dinov2 --mixin_config large --distributed_nproc_per_node "${GPU_NUM}" --distributed_do_spawn_workers --disable_wandb --weight_path $weight_path --output_dir="$output_dir" |& tee -a "$output_dir/train_stdout-$timestamp.log"