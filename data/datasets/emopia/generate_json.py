import pandas as pd
import json

# CSVファイルを読み込む
csv_file = "/home/Nakata/Music-Tri-Modal/data/datasets/emopia/label.csv"  # CSVファイルのパスを指定
df = pd.read_csv(csv_file)

# クラスを判定する関数を定義
def classify(q):
    if q == 1:
        return "Joy"
    elif q == 2:
        return "Anger"
    elif q == 3:
        return "Sadness"
    elif q == 4:
        return "Calmness"
    else:
        return None  # その他のケース

# クラス列を生成
df['class'] = df.apply(lambda row: classify(row['4Q']), axis=1)

# 必要な列を抽出してJSONフォーマットに変換
json_data = [
    {
        "audio_id": int(row['id']),
        "class": row['class'],
        "midi_file": row['ID']
    }
    for _, row in df.iterrows() if row['class'] is not None
]

# JSONデータを出力
output_file = "dataset_test.json"
with open(output_file, "w") as f:
    json.dump(json_data, f, indent=4)

print(f"JSONデータを{output_file}に保存しました。")
