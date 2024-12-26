import torch
import json
from tqdm import tqdm
import glob
from muscall.datasets.model import CP
import numpy as np
import logging

# ログファイルの設定
logging.basicConfig(
    filename='/home/Nakata/Music-Tri-Modal/error_log.txt',  # ログファイルのパス
    level=logging.ERROR,  # エラーレベル
    format='%(asctime)s - %(levelname)s - %(message)s'
)

data = glob.glob('/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/lmd_full/**/*.mid', recursive=True)

pbar = tqdm(data, disable=False, leave=True)
cp = CP(dict="/home/Nakata/Music-Tri-Modal/muscall/modules/MidiBERT/CP_origin.pkl")

files = []

for obj in pbar:
    try:
        # データの準備
        all_words = cp.prepare_data(midi_paths=[obj], task="", max_len=512)
        npy_file_path = obj.replace('.mid', '.npy')
        
        # データ保存
        np.save(npy_file_path, all_words)
        
    except Exception as e:
        # エラーが発生した場合はログに記録
        error_message = f"Error processing file {obj}: {e}"
        logging.error(error_message)
        print(error_message)  # コンソールにもエラーを表示
