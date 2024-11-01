# トークン化の可否をチェック
"""
from muscall.datasets.model import CP
import glob

cp = CP(dict="/home/Nakata/Music-Tri-Modal/muscall/modules/MidiBERT/CP.pkl")

files = glob.glob("/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/**/*.mid", recursive=True)

for i in range(100):
  all_words = cp.prepare_data(midi_paths=[files[i]], task="", max_len=512)
  print(all_words.shape)

"""

# lmd_matched_mp3の重複チェック1
"""
import os
from collections import defaultdict

def find_duplicate_files(directory):
    file_dict = defaultdict(list)
    
    # ディレクトリを再帰的に探索
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.mp3'):
                # ファイル名をキーとして保存し、パスをリストに追加
                file_dict[file].append(os.path.join(root, file))
    
    # 重複ファイルのチェック
    duplicates = {file: paths for file, paths in file_dict.items() if len(paths) > 1}
    
    # 結果を表示
    if duplicates:
        print("重複しているファイル:")
        for file, paths in duplicates.items():
            print(f"ファイル名: {file}")
            for path in paths:
                print(f"  - パス: {path}")
    else:
        print("重複するファイルは見つかりませんでした。")

# 使用例
find_duplicate_files('/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/audio/lmd_matched_mp3')  # 'A' ディレクトリを探索
"""

# lmd_matched_mp3の重複チェック2
"""
import os
import collections

files = glob.glob('/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/audio/lmd_matched_mp3/**/*.mp3', recursive=True)

file_names = []
for file in files:

    file_name = os.path.basename(file)
    file_names.append(file_name)

#print(collections.Counter(file_names))
print(print([k for k, v in collections.Counter(file_names).items() if v > 1]))
"""

# lp-musiccaps-MSDデータセット

import os
import json
import glob
from datasets import load_dataset
from tqdm import tqdm

# msd_datasetを辞書化
msd_dataset = load_dataset("seungheondoh/LP-MusicCaps-MSD")
mode = ["train", "valid", "test"]

# 辞書を作成
track_id_to_caption = {}
for m in mode:
    for item in msd_dataset[m]:
        track_id_to_caption[item["track_id"]] = item["caption_writing"]

# ファイルのリストを取得
files = glob.glob('/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/audio/lmd_matched_mp3/**/*.mp3', recursive=True)

data = []
for audio_id, file in enumerate(tqdm(files, desc="Processing files")):
    # ファイルIDを取得
    id = os.path.basename(file).replace(".mp3", "")
    
    # キャプションを取得
    caption = track_id_to_caption.get(id, "")

    # キャプションが無い場合
    if caption == "":
        aaa
    
    # オーディオパスを設定
    audio_path = file.replace("/audio/lmd_matched_mp3", "/midi/audio2midi").replace(".mp3", ".mid")
    
    # データをリストに追加
    data.append({"audio_id": audio_id, "caption": caption, "audio_path": audio_path})

# データをJSON形式に変換して保存
with open('dataset_all.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)