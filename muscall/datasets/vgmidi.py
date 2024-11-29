import os
import json
import numpy as np

import torch
from torch.utils.data.dataset import Dataset

from muscall.datasets.model import CP
import glob

class VGMIDI(Dataset):
    def __init__(self, config, dataset_type="test", midi_dic=None, data_root=None):
        super().__init__()
        if config is None:
            config = {}
        self.config = config # audiocaption.yamlの内容
        self._dataset_type = dataset_type
        self._data_dir = data_root

        self.CP = CP(dict=midi_dic)

        # データセットのJSONファイル(root/data/dataset/○○/dataset_○○.json)のパス
        self.dataset_json = os.path.join(self._data_dir, "dataset_{}.json".format(self._dataset_type))

        # audiocaption.yamlの内容
        self.midi_size = 8 # input_midiのtorch.Size([x, 512, 4])におけるxのサイズ
        self._load()

    # JSONファイルからデータを読み込み、音声ID、キャプション、音声パス、midiパスをリストに格納
    def _load(self):
        with open(self.dataset_json) as f:
            self.samples = json.load(f) # jsonをPythonオブジェクトとして読み込み

            self.audio_ids = [i["audio_id"] for i in self.samples] # jsonの各オブジェクトの"audio_id"(自然数)をリストに格納
            self.classes = [i["class"].strip() for i in self.samples] # jsonの各オブジェクトの"caption"をリストに格納
            self.midi_paths = [os.path.join(self._data_dir, i["midi_file"]) for i in self.samples] # jsonの各オブジェクトの"audio_path"(音声ファイルパス)をリストに格納

    def get_label(self, idx):
        return self.classes[idx]
    
    @torch.no_grad()
    def midi_padding(self, input_midi, idx):

        first_input_midi_shape = input_midi.shape[0]
        if input_midi.shape == torch.Size([0]):
            print(input_midi, self.midi_paths[idx])

        # パディング
        if input_midi.shape[0] < self.midi_size:
            x = self.midi_size - input_midi.shape[0]
            if isinstance(input_midi, np.ndarray):
                input_midi = torch.tensor(input_midi)
            input_midi = torch.nn.functional.pad(input_midi, (0, 0, 0, 0, 0, x))

        # クロップ
        elif input_midi.shape[0] > self.midi_size:
            input_midi = input_midi[:self.midi_size, :, :]

        return input_midi, first_input_midi_shape

    # 1つの曲の複数midiデータを取得し、すべてトークン化
    def get_midi(self, idx):
        files = [self.midi_paths[idx]]
        all_words = self.CP.prepare_data(files, task="", max_len=512) 
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
