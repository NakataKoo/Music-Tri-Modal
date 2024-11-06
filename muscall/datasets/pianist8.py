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
        """
        input_midi: (midi_size, 512, 4)
        512 → max token
        4 → Bar, Position, Pitch, Duration
        """

        first_input_midi_shape = input_midi.shape[0]
        if input_midi.shape == torch.Size([0]):
            print(input_midi, self.midi_dir_paths[idx])

        # パディング
        if input_midi.shape[0] < self.midi_size:
            x = self.midi_size - input_midi.shape[0]
            # 最初の次元にパディングを追加する
            # (0, 0) は 512 と 4 次元にはパディングを追加しないことを意味します
            input_midi = torch.nn.functional.pad(input_midi, (0, 0, 0, 0, 0, x))

        # クロップ
        elif input_midi.shape[0] > self.midi_size:
            input_midi = input_midi[:self.midi_size, :, :]

        return input_midi, first_input_midi_shape

    # 1つの曲の複数midiデータを取得し、すべてトークン化
    def get_midi(self, idx):
        #files = glob.glob(self.midi_dir_paths[idx]+'/*.mid', recursive=True) # ["path_to_midi0", "path_to_midi1","path_to_midi2", ...]
        files = [self.midi_paths[idx]]
        all_words = self.CP.prepare_data(files, task="", max_len=512) 
        '''
        all_wordsリストに、1つの曲のMIDIデータのトークン化されたデータが格納されている。
        all_wordsの各要素はリストで、これは1つのMIDIをスライシングした後に、それぞれトークン化し、
        それを複数MIDIに適用したもの。all_words=[[slice_words[0]], [slice_words[1]], ...], slice_words=[[[token0], [token1], ...], [[token0], [token1], ...], ...]
        
        len(all_words)が1曲分のエンベディング平均化時の「分母」となる
        '''
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
        return 8
