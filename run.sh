# Description: Run the training script.
# CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=gpu train.py  # ddim only for sample.


# Sampling with disagreement multi-rater labels.
CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=gpu sample.py --ddim True #  ddim only for sample.