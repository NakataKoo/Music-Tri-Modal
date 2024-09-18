import os
import json
import numpy as np

import torch
from torch.utils.data.dataset import Dataset

from muscall.datasets.model import CP
import glob

class Pianist8(Dataset):
    def __init__(self, config, dataset_type="test", midi_dic=None, data_root=None):
        super().__init__()
        if config is None:
            config = {}
        self.config = config # audiocaption.yamlの内容
        self._dataset_type = dataset_type
        self._data_dir = data_root # ${env.data_root}/datasets/${dataset_name}

        self.CP = CP(dict=midi_dic)

        # データセットのJSONファイル(root/data/dataset/○○/dataset_○○.json)のパス
        self.dataset_json = os.path.join(self._data_dir, "dataset_{}.json".format(self._dataset_type))

        # audiocaption.yamlの内容
        self.midi_size = self.config.dataset_config.midi.size_dim0 # input_midiのtorch.Size([x, 512, 4])におけるxのサイズ
        self._load()

    # JSONファイルからデータを読み込み、音声ID、キャプション、音声パス、midiパスをリストに格納
    def _load(self):
        with open(self.dataset_json) as f:
            self.samples = json.load(f) # jsonをPythonオブジェクトとして読み込み
            self.midi_dir = os.path.join(self._data_dir, "midi") # ${env.data_root}/datasets/${dataset_name}/midi

            self.audio_ids = [i["audio_id"] for i in self.samples] # jsonの各オブジェクトの"audio_id"(自然数)をリストに格納
            self.classes = [i["class"].strip() for i in self.samples] # jsonの各オブジェクトの"caption"をリストに格納
            self.midi_paths = [os.path.join(self.midi_dir, i["midi_file"]) for i in self.samples] # jsonの各オブジェクトの"audio_path"(音声ファイルパス)をリストに格納

    def get_label(self, idx):
        return self.classes[idx]
    
    @torch.no_grad()
    def midi_padding(self, input_midi, idx):

        first_input_midi_shape = input_midi.shape[0]
        if input_midi.shape == torch.Size([0]):
            print(input_midi, self.midi_paths[idx])

        # MIDI データの x 次元を最大サイズに揃える
        if input_midi.shape[0] < self.midi_size:
            # パディングして長さを合わせる
            midi_padding = torch.zeros((self.midi_size - input_midi.shape[0], 512, 4), dtype=input_midi.dtype)
            input_midi = torch.cat((input_midi, midi_padding), dim=0)

        #print(f"input_midiのshape: {input_midi.shape}")

        return input_midi, first_input_midi_shape

    @torch.no_grad()
    # 1つの曲の複数midiデータを取得し、すべてトークン化
    def get_midi(self, idx):
        #print(self.midi_paths[idx])
        all_words = self.CP.prepare_data([self.midi_paths[idx]], task="", max_len=512) 
        '''
        all_wordsリストに、1つの曲のMIDIデータのトークン化されたデータが格納されている。
        all_wordsの各要素はリストで、これは1つのMIDIをスライシングした後に、それぞれトークン化し、それを複数MIDIに適用したもの。all_words=[[slice_words[0]], [slice_words[1]], ...], slice_words=[[[token0], [token1], ..., [token512]], [[token0], [token1], ..., [token512]], ...]
        
        len(all_words)が1曲分のエンベディング平均化時の「分母」となる
        '''
        
        all_words = torch.from_numpy(all_words.astype(np.float32)).clone()
        all_words, first_input_midi_shape = self.midi_padding(all_words, idx)
        return all_words, first_input_midi_shape

    def __getitem__(self, idx):

        label = self.get_label(idx) # クラスラベルを取得
        input_midi, first_input_midi_shape = self.get_midi(idx) # midiデータを取得
        idx = torch.tensor(idx)
        return (
            label,
            input_midi,
            first_input_midi_shape,
            idx,
        )

    def __len__(self):
        return len(self.samples)

    @classmethod
    def num_classes(cls):
        return 4
