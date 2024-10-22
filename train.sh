export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
#export NCCL_SHM_DISABLE=1
export NCCL_DEBUG=INFO
#export NCCL_IB_DISABLE=1
#python scripts/train.py 2> warnings.txt \
python scripts/train.py \
--devices="0, 1"