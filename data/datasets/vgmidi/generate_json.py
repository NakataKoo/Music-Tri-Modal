import pandas as pd
import json

# CSVファイルを読み込む
csv_file = "/home/Nakata/Music-Tri-Modal/data/datasets/vgmidi/vgmidi_labelled.csv"  # CSVファイルのパスを指定
df = pd.read_csv(csv_file)

# クラスを判定する関数を定義
def classify(valence, arousal):
    if valence == 1 and arousal == 1:
        return "Joy"
    elif valence == -1 and arousal == 1:
        return "Anger"
    elif valence == -1 and arousal == -1:
        return "Sadness"
    elif valence == 1 and arousal == -1:
        return "Calmness"
    else:
        return None  # その他のケース

# クラス列を生成
df['class'] = df.apply(lambda row: classify(row['valence'], row['arousal']), axis=1)

# 必要な列を抽出してJSONフォーマットに変換
json_data = [
    {
        "audio_id": int(row['id']),
        "class": row['class'],
        "midi_file": row['midi']
    }
    for _, row in df.iterrows() if row['class'] is not None
]

# JSONデータを出力
output_file = "output.json"
with open(output_file, "w") as f:
    json.dump(json_data, f, indent=4)

print(f"JSONデータを{output_file}に保存しました。")
