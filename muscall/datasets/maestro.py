import os
import json
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from pytorch_memlab import profile

from muscall.datasets.model import CP
import glob
import librosa

class MAESTRO(Dataset):
    def __init__(self, config, tokenizer=None, midi_dic=None):

        super().__init__()
        self._dataset_name = "maestro"
        self._data_dir = self.config.data_dir # ${env.data_root}/datasets/${dataset_config.dataset_name}

        self.CP = CP(dict=midi_dic)

        # audiocaption.yamlの内容
        self.sample_rate = 48000 # サンプリングレート
        self.crop_length = 30 # 曲の長さ（秒）
        self.num_samples = self.sample_rate * self.crop_length # サンプリング数
        self.offset = 0.1
        self.random_crop = True # ランダムクロップの有無
        self.midi_size = 8 # input_midiのtorch.Size([x, 512, 4])におけるxのサイズ
        self._load()

    # JSONファイルからデータを読み込み、音声ID、キャプション、音声パス、midiパスをリストに格納
    def _load(self):

        self._data_dir

        files1 = glob.glob(self._data_dir + "maestro-testset/*.midi", recursive=True)
        files2 = glob.glob(self._data_dir + "maestro-testset/*.wav", recursive=True)
        files = list(set(file.split('.')[0] for file in (files1+files2)))
        
        self.audio_paths = [i+".wav" for i in files]
        self.midi_dir_paths = [i+".midi" for i in files]


    def get_audio(self, idx):

        audio, _ = librosa.load(self.audio_paths[idx], sr=self.sample_rate, mono=True, offset=self.offset, duration=self.crop_length)
        audio = audio.reshape(1, -1)
        
        # 短い音声に対しパディング
        if audio.shape[1] < self.num_samples:
            x = self.num_samples - audio.shape[1]
            padded_audio = np.pad(audio[0], ((0, x)), "mean")
            audio = np.array([padded_audio])
        # 長い音声に対しクロップ
        elif audio.shape[1] > self.num_samples:
            cropped_audio = audio[0][:self.num_samples]
            audio = np.array([cropped_audio])
        
        clipped_audio = np.clip(audio[0], -1.0, 1.0)
        clipped_audio = 2 * (clipped_audio - np.min(clipped_audio)) / (np.max(clipped_audio) - np.min(clipped_audio)) - 1
        audio = np.array([clipped_audio])
        
        audio = torch.from_numpy(audio.astype(np.float32)).clone()
        return audio
    
    def midi_padding(self, input_midi, idx):

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
        all_words = np.load(self.midi_dir_paths[idx]+".npy")
        all_words = torch.from_numpy(all_words.astype(np.float32)).clone()
        all_words, first_input_midi_shape = self.midi_padding(all_words, idx)
        return all_words, first_input_midi_shape

    def __getitem__(self, idx):
        audio_id = "" # "audio_id"

        input_audio = self.get_audio(idx) # 音声データを取得
        input_text = "" # キャプションを取得
        input_midi, first_input_midi_shape = self.get_midi(idx) # midiデータを取得

        idx = torch.tensor(idx)

        return (
            audio_id,
            input_audio,
            input_text,
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
