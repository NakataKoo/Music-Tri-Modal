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

at Music-Tri-Modal/
```
wget https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt
```

## Preparing the dataset
MusCALL is trained on a multimodal dataset of (audio, text) pairs. 

Annotations should be provided in JSON format and must include the following fields:

```audio_id```:     the unique identifier for each audio track in the dataset

```caption``` :     a string with the textual description of the audio track 

```audio_path```:   path to the audio track, relative to the root audio directory

One JSON file per split must be provided and stored in the [`data/datasets`](data/datasets/) directory, following this structure:

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

An illustrative example of the dataset is provided in [`data/datasets/audiocaption/`](data/datasets/audiocaption/).

### Midi-Audio Data prepare
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
python evaluate.py <model_id> retrieval
```

or, in the zero-shot transfer setting, on an arbitrary music classification task.

In our zero-shot evaluation, we include:

* `mtt`: auto-tagging on the [MagnaTagATune Dataset](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset)
* `gtzan`: music genre classification on the [GTZAN dataset](http://marsyas.info/downloads/datasets.html)

```bash
python evaluate.py <model_id> zeroshot <dataset_name>
```

You'll need to download the datasets inside the [`datasets/`](datasets/) folder and preprocess them before running the zeroshot evaluation.

## Reference
This repository is based on https://github.com/ilaria-manco/muscall
