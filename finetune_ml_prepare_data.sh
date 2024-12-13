export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
export PYTHONWARNINGS="ignore"
export CUDA_VISIBLE_DEVICES="0"
python scripts/finetune_ml_prepare_data.py \
--checkpoint="" \
--dataset="piansit8" \
--midi_size=10