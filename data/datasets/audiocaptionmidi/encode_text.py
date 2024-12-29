import laion_clap
import torch
import os
from tqdm import tqdm
import json
import time

@torch.no_grad()
def encode_text(clap, text, text_mask=None):
    
    text_features = clap.get_text_embedding(text, use_tensor=True)
    
    text_features = text_features.to(device)
    return text_features

device = torch.device("cuda")

clap = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base', device=device)
clap.load_ckpt("/raid/m236866/Music-Tri-Modal/ckpt/music_audioset_epoch_15_esc_90.14.pt", verbose=False)

for param in clap.parameters():
    param.requires_grad = False

# JSONファイルを読み込む
file_path = "dataset_all.json"
output_path = "dataset_all_with_caption_emb.json"

with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# `tqdm`を使用して進捗バーを表示しながらループを実行
for idx, item in enumerate(tqdm(data, desc="Processing captions"), start=1):

    text = [item['caption']]
    text_emb = encode_text(clap, text)
    item['caption_embedding'] = text_emb.tolist()[0]

with open(output_path, 'w', encoding='utf-8') as output_file:
    json.dump(data, output_file, indent=4, ensure_ascii=False)