## Setup
Create a fresh virtual environment:

```setup
python -m venv venv 
source venv/bin/activate
```

Then, clone the repository and install the dependencies:
(研究室のA40サーバーの場合)
```setup
git clone https://www.github.com/ilaria-manco/muscall 
cd muscall 
pip install -r requirements.txt
pip install -e .
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install laion_clap
pip install miditoolkit
```

at Music-Tri-Modal/ckpt/
```
wget https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt
```

## Preparing the dataset

### MIDI-Text-Audio Pair ver

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

### MIDI-Text Pair ver

The Music-Tri-Modal is trained on a multimodal dataset of (audio, text, midi) pairs. 

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

An illustrative example of the dataset is provided in [`data/datasets/audiocaption/`](data/datasets/audiocaption/).

```
wget https://huggingface.co/datasets/amaai-lab/MidiCaps/resolve/main/train.json
wget https://huggingface.co/datasets/amaai-lab/MidiCaps/resolve/main/midicaps.tar.gz
tar -zxvf midicaps.tar.gz
```

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

## Training
Dataset, model and training configurations are set in the respective `yaml` files in [`configs`](configs). You can also pass some options via the CLI, overwriting the arguments in the config files. For more details on the CLI options, please refer to the [training script](scripts/train.py).

To train the model with the default configs, simply run

```bash
cd scripts/
python train.py 
```

This will generate a `model_id` and create a new folder in [`save/experiments/`](save/experiments/) where the output will be saved.

If you wish to resume training from a saved checkpoint, run this command:

```bash
python train.py --experiment_id <model_id> 
```

## Evaluating
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
python evaluate.py --experiment_id <model_id> zeroshot --dataset_name <dataset_name>
```

You'll need to download the datasets inside the [`datasets/`](datasets/) folder and preprocess them before running the zeroshot evaluation.

## Fine-Tuning

```bash
python finetune.py --experiment_id <model_id> --dataset <dataset_name>
```

## Reference
This repository is based on https://github.com/ilaria-manco/muscall
