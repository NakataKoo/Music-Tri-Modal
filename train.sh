export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
#export NCCL_SHM_DISABLE=1
# export NCCL_DEBUG=INFO
#export NCCL_IB_DISABLE=1
#export NCCL_TMP_DIR=/home/Nakata/nccl_tmp
export PYTHONWARNINGS="ignore"
#export CUDA_LAUNCH_BLOCKING=1
# python scripts/train.py 2> result.txt \
export CUDA_VISIBLE_DEVICES="1"
python scripts/train.py