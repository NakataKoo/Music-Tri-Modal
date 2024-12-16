import os
from tqdm import tqdm
import numpy as np
import json
import time
import logging
from sklearn.model_selection import train_test_split
from muscall.datasets.model import CP

# ログファイルの設定
logging.basicConfig(
    filename='error_log.txt',  # ログファイルのパス
    level=logging.ERROR,  # エラーレベル
    format='%(asctime)s - %(levelname)s - %(message)s'
)

cp = CP(dict="/home/Nakata/Music-Tri-Modal/muscall/modules/MidiBERT/CP.pkl")

# JSONファイルを読み込む
file_path = "dataset_all.json"

with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

all_data = []
all_ans = []

# `tqdm`を使用して進捗バーを表示しながらループを実行
for idx, item in enumerate(tqdm(data, desc="Processing captions"), start=1):
    midi = item['midi_file']
    label = item['class']

    try:
        # データの準備
        all_words = cp.prepare_data(midi_paths=["midi/"+midi+".mid"], task="", max_len=512)
        all_words = all_words.tolist()
        all_data.append(all_words)

        # ラベルの設定
        if label == "Joy":
            label_num = 0
        elif label == "Anger":
            label_num = 1
        elif label == "Sadness":
            label_num = 2
        elif label == "Calmness":
            label_num = 3
        else:
            raise ValueError(f"Unknown label: {label}")
        
        all_ans.append(label_num)

    except Exception as e:
        # エラーが発生した場合はログに記録
        error_message = f"Error processing file {item}: {e}"
        logging.error(error_message)
        print(error_message)  # コンソールにもエラーを表示

# データ保存
all_data = np.array(all_data)
all_ans = np.array(all_ans)

np.save("/home/Nakata/Music-Tri-Modal/data/datasets/emopia/emopia.npy", all_data)
np.save("/home/Nakata/Music-Tri-Modal/data/datasets/emopia/emopia_ans.npy", all_ans)

# データセットの分割
# クラスごとのインデックスを取得
unique_classes = np.unique(all_ans)
train_indices = []
val_indices = []
test_indices = []

for cls in unique_classes:
    cls_indices = np.where(all_ans == cls)[0]
    train_idx, temp_idx = train_test_split(cls_indices, test_size=0.3, random_state=42)  # 60% train
    val_idx, test_idx = train_test_split(temp_idx, test_size=(1/3), random_state=42)       # 20% val, 20% test
    train_indices.extend(train_idx)
    val_indices.extend(val_idx)
    test_indices.extend(test_idx)

# インデックスをシャッフル
train_indices = np.random.permutation(train_indices)
val_indices = np.random.permutation(val_indices)
test_indices = np.random.permutation(test_indices)

# データを分割
train_data = all_data[train_indices]
train_labels = all_ans[train_indices]

val_data = all_data[val_indices]
val_labels = all_ans[val_indices]

test_data = all_data[test_indices]
test_labels = all_ans[test_indices]

# 分割データを保存
np.save("/home/Nakata/Music-Tri-Modal/data/datasets/emopia/emopia_train.npy", train_data)
np.save("/home/Nakata/Music-Tri-Modal/data/datasets/emopia/emopia_train_ans.npy", train_labels)
np.save("/home/Nakata/Music-Tri-Modal/data/datasets/emopia/emopia_val.npy", val_data)
np.save("/home/Nakata/Music-Tri-Modal/data/datasets/emopia/emopia_val_ans.npy", val_labels)
np.save("/home/Nakata/Music-Tri-Modal/data/datasets/emopia/emopia_test.npy", test_data)
np.save("/home/Nakata/Music-Tri-Modal/data/datasets/emopia/emopia_test_ans.npy", test_labels)

print("Dataset preparation complete.")