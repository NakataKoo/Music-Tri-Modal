import json

# JSONファイルのパス
json_file_path = '/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/dataset_all_new.json'

# JSONファイルを読み込み
with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 各オブジェクトの「audio_path」をreplaceで変更
for obj in data:
  obj['audio_path'] = obj['audio_path'].replace('/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi', 'lmd_matched_mp3').replace(".mid", "")

# 変更した内容を元のJSONファイルに書き込み
with open(json_file_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print("audio_pathの文字列が変更され、JSONファイルに保存されました。")
