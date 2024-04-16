# Fix python linux setup
1) export PYTHONPATH=/home/duser/ba-gujen1-steij14:$PYTHONPATH

# Training on multiple GPUs
1) Set CUDA_VISIBLE_DEVICES to more than one GPU
2) torchrun --nproc_per_node 2 train.py