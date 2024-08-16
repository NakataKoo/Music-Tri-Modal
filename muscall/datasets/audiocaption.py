import os
import json
import numpy as np

import torch
from torch.utils.data.dataset import Dataset

from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.clip.tokenization_clip import CLIPTokenizer

from muscall.datasets.model import CP
import glob
import librosa

class AudioCaptionMidiDataset(Dataset):
    def __init__(self, config, tokenizer=None, dataset_type="train"):
        """Constructs an AudioCaptionMidiDataset dataset.

        Args:
        - config: (dict-like object) dataset config
        - tokenizer: (tokenizer object) default is BertTokenizer from transformers library
        - dataset_type: (String) "train", "test" or "val"
        """
        super().__init__()
        if config is None:
            config = {}
        self.config = config # audiocaption.yamlの内容
        self._dataset_name = "audio-caption-midi-pair"
        self._dataset_type = dataset_type
        self._data_dir = self.config.data_dir # ${env.data_root}/datasets/${dataset_config.dataset_name}

        self.CP = CP(dict="/content/Music-Tri-Modal/muscall/modules/MidiBERT/CP.pkl")

        self.dataset_json = os.path.join(
            self._data_dir, "dataset_{}.json".format(self._dataset_type)
        ) # データセットのJSONファイル(root/data/dataset/○○/dataset_○○.json)のパス

        # audiocaption.yamlの内容
        #self.max_seq_length = self.config.text.max_seq_length # 最大シーケンス長
        self.sample_rate = self.config.audio.sr # サンプリングレート
        self.num_samples = self.sample_rate * self.config.audio.crop_length # クロップ長
        self.random_crop = self.config.audio.random_crop # ランダムクロップの有無
        self._load()

    # JSONファイルからデータを読み込み、音声ID、キャプション、音声パス、midiパスをリストに格納
    def _load(self):
        with open(self.dataset_json) as f:
            self.samples = json.load(f) # jsonをPythonオブジェクトとして読み込み
            self.audio_dir = os.path.join(self._data_dir, "audio") # ${env.data_root}/datasets/${dataset_config.dataset_name}/audio
            self.midi_dir = os.path.join(self._data_dir, "midi") # ${env.data_root}/datasets/${dataset_config.dataset_name}/midi

            self.audio_ids = [i["audio_id"] for i in self.samples] # jsonの各オブジェくトの"audio_id"(自然数)をリストに格納
            self.captions = [i["caption"].strip() for i in self.samples] # jsonの各オブジェくトの"caption"をリストに格納
            self.audio_paths = [os.path.join(
                self.audio_dir, i["audio_path"]) for i in self.samples] # jsonの各オブジェくトの"audio_path"(音声ファイルパス)をリストに格納
            self.midi_dir_paths = [os.path.join(self.midi_dir, os.path.splitext(i["audio_path"])[0].replace('lmd_matched_mp3', 'lmd_aligned')) for i in self.samples] # 各楽曲のmidiファイルが格納されたディレクトリのパスをリストに格納

    def get_raw_caption(self, idx):
        """Get raw caption text"""

        return self.captions[idx]

    # 改良（音声データを読み込み、クロップし、テンソルに変換）
    def get_audio(self, idx):
        audio_path = self.audio_paths[idx]
        audio, sr = librosa.load(audio_path, sr=48000, mono=True)
        #audio = audio.reshape(1, -1)
        
        # 1次元のテンソルに変換
        # audio = torch.tensor(audio, dtype=torch.float)
        audio = torch.from_numpy(audio.astype(np.float32)).clone()

        # 音声が短い場合はゼロパディングし、長い場合はトリミング
        if len(audio) < self.num_samples:
            zeros_needed = torch.zeros(self.num_samples - len(audio))
            audio = torch.cat((audio, zeros_needed), dim=0)
        elif len(audio) > self.num_samples:
            audio = audio[:self.num_samples]
            
        # もしaudioがCUDAテンソルであれば、CPUに移動
        #if audio.is_cuda:
        #    audio = audio.cpu()
        #audio = audio.to('cpu').detach().numpy().copy()
        
        return audio
    
    # 1つの曲の複数midiデータを取得し、すべてトークン化
    def get_midi(self, idx):
        files = glob.glob(self.midi_dir_paths[idx]+'/*.mid', recursive=True) # ["path_to_midi0", "path_to_midi1","path_to_midi2", ...]
        all_words, _, _, _ = self.CP.prepare_data(files, task="", max_len=512) 
        '''
        all_wordsリストに、1つの曲のMIDIデータのトークン化されたデータが格納されている。
        all_wordsの各要素はリストで、これは1つのMIDIをスライシングした後に、それぞれトークン化し、それを複数MIDIに適用したもの。all_words=[[slice_words[0]], [slice_words[1]], ...], slice_words=[[[token0], [token1], ...], [[token0], [token1], ...], ...]
        
        len(all_words)が1曲分のエンベディング平均化時の「分母」となる
        '''
        
        all_words = torch.from_numpy(all_words.astype(np.float32)).clone()
        return all_words

    def __getitem__(self, idx):
        audio_id = torch.tensor(self.audio_ids[idx], dtype=torch.long) # audio_idsは"audio_id"リスト

        input_audio = self.get_audio(idx) # 音声データを取得
        input_text = self.get_raw_caption(idx) # キャプションを取得
        input_midi = self.get_midi(idx) # midiデータを取得

        idx = torch.tensor(idx)

        return (
            audio_id,
            input_audio,
            input_text,
            input_midi,
            idx,
        )

    def __len__(self):
        return len(self.samples)

    @classmethod
    def config_path(cls):
        return "configs/datasets/audiocaption.yaml"
