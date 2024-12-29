import os
import json
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from pytorch_memlab import profile

from muscall.datasets.model import CP
import glob

class AudioCaptionMidiDataset(Dataset):
    def __init__(self, config, tokenizer=None, dataset_type="train", midi_dic=None):

        super().__init__()
        if config is None:
            config = {}
        self.config = config # audiocaption.yamlの内容
        self._dataset_name = "audio-caption-midi-pair"
        self._dataset_type = dataset_type
        self._data_dir = self.config.data_dir # ${env.data_root}/datasets/${dataset_config.dataset_name}

        self.CP = CP(dict=midi_dic)

        # データセットのJSONファイル(root/data/dataset/○○/dataset_○○.json)のパス
        self.dataset_json = os.path.join(self._data_dir, "dataset_{}.json".format(self._dataset_type))

        # audiocaption.yamlの内容
        self.midi_size = self.config.midi.size_dim0 # input_midiのtorch.Size([x, 512, 4])におけるxのサイズ
        self._load()

    # JSONファイルからデータを読み込み、音声ID、キャプション、音声パス、midiパスをリストに格納
    def _load(self):
        with open(self.dataset_json) as f:
            self.samples = json.load(f) # jsonをPythonオブジェクトとして読み込み

            self.audio_ids = [i["audio_id"] for i in self.samples]
            self.caption_enbs = [i["caption_embedding"] for i in self.samples]
            self.midi_dir_paths = [os.path.join(self._data_dir, os.path.splitext(i["audio_path"])[0].replace(".mid", ".npy")) for i in self.samples]

    def get_caption_embedding(self, idx):
        """Get text embedding"""
        return torch.tensor(self.caption_enbs[idx])
    
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
        #files = [self.midi_dir_paths[idx]+".mid"]
        #all_words = self.CP.prepare_data(files, task="", max_len=512) 
        '''
        all_wordsリストに、1つの曲のMIDIデータのトークン化されたデータが格納されている。
        all_wordsの各要素はリストで、これは1つのMIDIをスライシングした後に、それぞれトークン化し、
        それを複数MIDIに適用したもの。all_words=[[slice_words[0]], [slice_words[1]], ...], slice_words=[[[token0], [token1], ...], [[token0], [token1], ...], ...]
        
        len(all_words)が1曲分のエンベディング平均化時の「分母」となる
        '''
        all_words = np.load(self.midi_dir_paths[idx]+".npy")
        all_words = torch.from_numpy(all_words.astype(np.float32)).clone()
        all_words, first_input_midi_shape = self.midi_padding(all_words, idx)
        return all_words, first_input_midi_shape

    def __getitem__(self, idx):
        audio_id = torch.tensor(self.audio_ids[idx], dtype=torch.long) # "audio_id"

        input_text_enb = self.get_caption_embedding(idx) # キャプションを取得
        input_midi, first_input_midi_shape = self.get_midi(idx) # midiデータを取得

        idx = torch.tensor(idx)

        return (
            audio_id,
            input_text_enb,
            input_midi,
            first_input_midi_shape,
            self.midi_dir_paths,
            idx,
        )

    def __len__(self):
        return len(self.samples)

    @classmethod
    def config_path(cls):
        return "configs/datasets/audiocaption.yaml"
