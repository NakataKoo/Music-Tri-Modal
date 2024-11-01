import torch
import json
from tqdm import tqdm
from muscall.datasets.model import CP

# JSONファイルのパス
json_file_path = '/home/Nakata/lp-music-caps/lpmc/music_captioning/dataset_all.json'

# JSONファイルを読み込み
with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 各オブジェクトの「audio_path」をreplaceで変更
data_new = []
del_data = []
pbar = tqdm(data, disable=False, leave=True)
cp = CP(dict="/home/Nakata/Music-Tri-Modal/muscall/modules/MidiBERT/CP.pkl")
for obj in pbar:
  all_words = cp.prepare_data(midi_paths=[obj['audio_path']], task="", max_len=512) 
  if all_words.shape == torch.Size([0]):
    print("del: "+obj['audio_path'])
    del_data.append({"audio_path": obj['audio_path']})
    continue
  data_new.append(obj)

# 変更した内容を元のJSONファイルに書き込み
with open("dataset_all_new.json", 'w', encoding='utf-8') as file:
    json.dump(data_new, file, ensure_ascii=False, indent=4)

with open("deleted_midi_data.json", 'w', encoding='utf-8') as file:
    json.dump(del_data, file, ensure_ascii=False, indent=4)

print("eventの無いmidiファイルを削除したデータセットを、JSONファイルに保存されました。")