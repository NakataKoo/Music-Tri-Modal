## 学習で使用するデータセット

configs/datasets/audiocaption.yaml内の「dataset_name」の値を、使用したいデータセット名に変更する。

### data/datasets/のフォルダ構成

data/datasets/dataset_nameのように「dataset_name」のようにフォルダを作成し、以下の構成でデータを用意する。

```
dataset_name
├── audio            
│   ├── track_1.npy
│   ├── track_2.npy
|   └── ...
├── dataset_train.json    
├── dataset_val.json    
└── dataset_test.json
```

Tri-modal is trained on a multimodal dataset of (audio, text, midi) pairs. 

Annotations should be provided in JSON format and must include the following fields:

```audio_id```:     the unique identifier for each audio track in the dataset（1から始まる自然数の昇順で良い。例. 1, 2, 3, 4...）

```caption``` :     a string with the textual description of the audio track（いわゆるテキストキャプション） 

```audio_path```:   path to the audio track, relative to the root audio directory （音楽データが格納されているファイルパス。ファイル名で良い）

## wavやmp3をnpyに変換

wavやmp3などの音声ファイルを、あらかじめnpyに変換する必要がある。

npyファイルは、Pythonのnumpyライブラリによって作成されるバイナリファイル形式で、numpy配列を効率的に保存および読み込むために使用されます。npyファイルは、数値データを高速に保存・読み込みするためのフォーマットであり、データの形状やデータ型などの情報も一緒に保存されます。

音声ファイルをnpyファイルに変換するには、まず音声ファイルを読み込み、次にそのデータをnumpy配列として保存する必要があります。以下にそのためのPythonコードを示します。

まず、必要なライブラリをインストールします。以下のコマンドを使用してください。

```bash
pip install numpy soundfile
```

次に、以下のPythonスクリプトを作成してください。このスクリプトは指定されたフォルダ内のすべての音声ファイルをnpyファイルに変換します。

```python
import os
import numpy as np
import soundfile as sf

def convert_audio_to_npy(input_folder, output_folder):
    # 入力フォルダ内のすべてのファイルを再帰的に取得
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith('.wav') or filename.endswith('.mp3'):
                # 音声ファイルのパスを取得
                file_path = os.path.join(root, filename)
                # 音声ファイルを読み込む
                data, samplerate = sf.read(file_path)
                # 出力ファイルのパスを作成
                relative_path = os.path.relpath(root, input_folder)
                output_subfolder = os.path.join(output_folder, relative_path)
                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder)
                output_file_path = os.path.join(output_subfolder, os.path.splitext(filename)[0] + '.npy')
                # データをnpyファイルとして保存
                np.save(output_file_path, data)
                print(f"Converted {file_path} to {output_file_path}")

input_folder = 'lmd_matched_mp3'
output_folder = 'output_folder'

# 出力フォルダが存在しない場合は作成
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

convert_audio_to_npy(input_folder, output_folder)
```

```input_folder```：変換したい音声ファイルが格納されているフォルダのパスを指定します。

```output_folder```：変換後のnpyファイルを保存するフォルダのパスを指定します。

スクリプトを実行すると、指定されたフォルダ内のすべての音声ファイルがnpyファイルに変換され、指定された出力フォルダに保存されます。