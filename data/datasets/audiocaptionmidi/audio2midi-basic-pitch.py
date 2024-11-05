import subprocess
import json
from tqdm import tqdm

# JSONファイルを読み込み
with open("/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/deleted_midi_data.json", 'r', encoding='utf-8') as file:
    data = json.load(file)

pbar = tqdm(data, disable=False, leave=True)
for obj in pbar:
    audio_path = obj['audio_path'].replace("/midi/audio2midi", "/audio/lmd_matched_mp3").replace(".mid", ".mp3")
    command = "basic-pitch /home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/basic-pitch-output "+audio_path
  
    # コマンドを実行し、出力を取得
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        print("Command output:\n", result.stdout)  # コマンドの出力を表示
    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)  # エラーメッセージを表示
