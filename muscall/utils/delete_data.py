import pandas as pd
import os
import shutil
import numpy as np

# ファイル名を指定してください
file_name = '/home_lab/nakata/Music-Tri-Modal/not_midis.npy'

# .npyファイルを読み込みます
data = np.load(file_name)
not_folders = data.tolist()

# 特定のディレクトリのパス
target_directory = "/home_lab/nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/lmd_aligned"

# リストAのファイルを削除
for file_path in not_folders:
    file_path = "/home_lab/nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/" + file_path
    # ファイルが存在するか確認
    if os.path.exists(file_path) and file_path.startswith(target_directory):
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except OSError as e:
            print(f"Error deleting {file_path}: {e}")
    else:
        print(f"File not found or outside target directory: {file_path}")

def remove_empty_dirs(directory):
    # ディレクトリ内を再帰的に走査
    for root, dirs, files in os.walk(directory, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            # ディレクトリが空かどうかを確認
            if not os.listdir(dir_path):
                # 空のディレクトリを削除
                os.rmdir(dir_path)
                print(f"Deleted empty directory: {dir_path}")

# 空のサブディレクトリを削除
remove_empty_dirs(target_directory)