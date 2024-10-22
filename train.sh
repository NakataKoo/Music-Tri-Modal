export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
#python scripts/train.py 2> warnings.txt \
python scripts/train.py \
--devices=0
