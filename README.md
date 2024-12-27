# 1. Setup
Create a fresh virtual environment:

## 1-1. Create Venv
Python Venv:
```setup
python -m venv venv 
source venv/bin/activate
```

Anaconda:
```setup
conda create -n venv python=3.9 jupyter
conda activate venv
```

## 1-2. Install packages

Then, clone the repository and install the dependencies(Python 3.9, CUDA=12.1):
```setup
pip install -r requirements.txt
pip install -e .
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install laion_clap
pip install miditoolkit
```

## 1-3. Download CLAP checkpoint

at ```Music-Tri-Modal/ckpt/```
```
wget https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt
```

# 2. Preparing the dataset

## 2-1. MIDI-Text-Audio Pair ver

The Music-Tri-Modal is trained on a multimodal dataset of (audio, text, midi) pairs. 

Annotations should be provided in JSON format and must include the following fields:

- ```audio_id```:     the unique identifier for each audio track in the dataset

- ```caption``` :     a string with the textual description of the audio track 

- ```audio_path```:   path to the audio track, relative to the root audio directory

One JSON file per split must be provided and stored in the [`data/datasets`](data/datasets/) directory, following this structure:

```
dataset_name
├── audio            
│   ├── track_1.mp3
│   ├── track_2.mp3
|   └── ...
├── midi
|   ├── track_1/
|   |   ├── track_1_1.mid
|   |   ├── track_1_2.mid
|   |   └── ...   
|   ├── track_2/
|   |   ├── track_2_1.mid
|   |   ├── track_2_2.mid
|   |   └── ...   
|   └── ...
├── dataset_train.json    
├── dataset_val.json    
└── dataset_test.json
```

An illustrative example of the dataset is provided in [`data/datasets/audiocaption/`](data/datasets/audiocaption/).

```
cd /Music-Tri-Modal/data/datasets/audiocaptionmidi/

mkdir midi
mkdir audio

cd midi
wget http://hog.ee.columbia.edu/craffel/lmd/lmd_aligned.tar.gz
tar -zxvf lmd_aligned.tar.gz

cd audio
wget http://hog.ee.columbia.edu/craffel/lmd/lmd_matched_mp3.tar.gz
tar -xzvf lmd_matched_mp3.tar.gz
```

## 2-2. MIDI-Text Pair ver

Bridge Alignment(Text-MIDI pair only) ver.

Annotations should be provided in JSON format and must include the following fields:

- ```audio_id```:     the unique identifier for each audio track in the dataset

- ```caption``` :     a string with the textual description of the audio track 

- ```audio_path```:   path to the audio track, relative to the root audio directory

One JSON file per split must be provided and stored in the [`data/datasets`](data/datasets/) directory, following this structure:

```
dataset_name
├── lmd_full
|   ├── 0/
|   |   ├── track_1_1.mid
|   |   ├── track_1_2.mid
|   |   └── ...   
|   ├── 1/
|   |   ├── track_2_1.mid
|   |   ├── track_2_2.mid
|   |   └── ...   
|   └── ...
├── dataset_train.json    
├── dataset_val.json    
└── dataset_test.json
```

### 2-2-1. Original Dataset 

An illustrative example of the dataset is provided in [`data/datasets/audiocaption/`](data/datasets/audiocaption/).

```
wget https://huggingface.co/datasets/amaai-lab/MidiCaps/resolve/main/train.json
wget https://huggingface.co/datasets/amaai-lab/MidiCaps/resolve/main/midicaps.tar.gz
tar -zxvf midicaps.tar.gz
```

To make ```dataset_all.json``` from "train.json" in [MidiCaps](https://huggingface.co/datasets/amaai-lab/MidiCaps), run bellow:

```python
import json
import os
import pandas as pd

with open("train.json", 'r', encoding='utf-8') as file:
    data = file.readlines()

midi_files = []
for i in range(len(data)):
    midi_files.append(json.loads(data[i])["location"])

captions = []
for i, midi_file in enumerate(midi_files):
    if os.path.exists(midi_file):
        captions.append(json.loads(data[i])["caption"])
    else:
        continue

# データフレームの作成
df = pd.DataFrame({
    'midi_file': midi_files,
    'caption': captions
})

# JSON形式の変換処理
json_data = []
for index, row in df.iterrows():
    json_data.append({
        "audio_id": index,  # 自然数の昇順
        "caption": row['caption'],  # CSVのcaption列
        "audio_path": row['midi_file']  # CSVのmidi_file列
    })

# JSONデータをファイルに保存
json_file_path = 'dataset_all.json'  # 出力するJSONファイルのパス
with open(json_file_path, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=4)

print(f"JSONデータが{json_file_path}に保存されました。")

```

### 2-2-2. My Dataset

In ```data/datasets/audiocaptionmidi/```

```bash
wget https://huggingface.co/datasets/KooNakata/Music-Tri-Modal/resolve/main/midicaps.tar.gz
tar -zxvf midicaps.tar.gz
```

# 3. Quick Start

- Training
- Evaluation
- Fine-Tune

## 3-1. Training
Dataset, model and training configurations are set in the respective `yaml` files in [`configs`](configs). You can also pass some options via the CLI, overwriting the arguments in the config files. For more details on the CLI options, please refer to the [training script](scripts/train.py).

To train the model with the default configs, simply run

```bash
source scripts/train.sh
```

This will generate a `model_id` and create a new folder in [`save/experiments/`](save/experiments/) where the output will be saved.

If you wish to resume training from a saved checkpoint, run this command:

```bash
python scripts/train.py --experiment_id <model_id> 
```

## 3-2. Evaluating
Once trained, you can evaluate Model on the cross-modal retrieval task:

```bash
python evaluate.py --experiment_id <model_id> retrieval
```

or, in the zero-shot transfer setting, on an arbitrary music classification task.

In our zero-shot evaluation, we include:

- Pianist8(Composer, class 8)
- EMOPIA(Emotion, class 4)
- VGMIDI(Emotion, class 4)
- WIKIMT(Genre, class 8)

```bash
python scripts/evaluate.py --experiment_id <model_id> zeroshot --dataset_name <dataset_name>
```

You'll need to download the datasets inside the [`datasets/`](datasets/) folder and preprocess them before running the zeroshot evaluation.

## 3-3. Fine-Tuning

```bash
source scripts/finetune.sh
```

In ```scripts/finetune.sh```:

```bash
python scripts/finetune.py --experiment_id <model_id> --dataset <dataset_name>
```

# 4. Caution

環境を変えて実行する場合、configファイルのパスを適宜変える必要がある。

# 5. Reference
This repository is based on https://github.com/ilaria-manco/muscall
