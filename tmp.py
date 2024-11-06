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

"""
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
"""

# データセットの音声ファイルの長さを取得

import os
from mutagen.mp3 import MP3
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import statistics
import numpy as np

def get_mp3_duration(file_path):
    """MP3ファイルの長さを取得する関数"""
    try:
        audio = MP3(file_path)
        return audio.info.length, file_path  # 長さとパスをタプルで返す
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def get_mp3_durations(directory, threshold):
    # 対象のmp3ファイルを収集
    mp3_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.mp3'):
                mp3_files.append(os.path.join(root, file))

    # tqdmで進捗を表示しながら並列処理を実行
    durations = []
    with ProcessPoolExecutor() as executor:
        # ファイルごとに非同期にMP3の長さを取得
        futures = {executor.submit(get_mp3_duration, file): file for file in mp3_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing MP3 files"):
            result = future.result()
            if result is not None:
                durations.append(result)  # タプル (length, file_path) を追加

    if durations:
        # 各ファイルの長さだけのリストを作成
        lengths = [duration[0] for duration in durations]
        
        # 最大・最小ファイルの特定
        max_duration_file = max(durations, key=lambda x: x[0])
        min_duration_file = min(durations, key=lambda x: x[0])
        
        # 統計値の計算
        avg_duration = statistics.mean(lengths)
        median_duration = statistics.median(lengths)
        q1_duration = np.percentile(lengths, 25)  # 第1四分位数
        q3_duration = np.percentile(lengths, 75)  # 第3四分位数
        iqr = q3_duration - q1_duration  # 四分位範囲

        # 外れ値のしきい値計算
        lower_bound = q1_duration - 1.5 * iqr
        upper_bound = q3_duration + 1.5 * iqr
        
        # 外れ値の特定
        outliers = [(duration, path) for duration, path in durations if duration < lower_bound or duration > upper_bound]

        # 指定した秒数を超えるファイルの数をカウント
        above_threshold = [(duration, path) for duration, path in durations if duration > threshold]

        # 結果を表示
        print(f"Total MP3 files found: {len(durations)}")
        print(f"Maximum duration: {max_duration_file[0]:.2f} seconds - File: {max_duration_file[1]}")
        print(f"Minimum duration: {min_duration_file[0]:.2f} seconds - File: {min_duration_file[1]}")
        print(f"Average duration: {avg_duration:.2f} seconds")
        print(f"Median duration: {median_duration:.2f} seconds")
        print(f"1st Quartile (Q1): {q1_duration:.2f} seconds")
        print(f"3rd Quartile (Q3): {q3_duration:.2f} seconds")
        print(f"IQR: {iqr:.2f} seconds")
        print(f"Lower bound for outliers: {lower_bound:.2f} seconds")
        print(f"Upper bound for outliers: {upper_bound:.2f} seconds")
        
        # 外れ値の表示
        if outliers:
            print("\nOutlier files (outside of IQR bounds):")
            for duration, path in outliers:
                print(f"Duration: {duration:.2f} seconds - File: {path}")
        else:
            print("\nNo outliers found based on the IQR method.")
        
        # 指定秒数を超えるファイルの数を表示
        print(f"\nNumber of files exceeding {threshold} seconds: {len(above_threshold)}")
        if above_threshold:
            print("Files exceeding the threshold:")
            for duration, path in above_threshold:
                print(f"Duration: {duration:.2f} seconds - File: {path}")
                pass
    else:
        print("No MP3 files found in the specified directory.")

# 使用例: 特定のディレクトリAを指定し、しきい値を設定
directory_a = '/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/audio/lmd_matched_mp3'
threshold_seconds = 65  # しきい値（秒）
get_mp3_durations(directory_a, threshold_seconds)