# Description: Run the training script.
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=gpu train.py  # ddim only for sample.