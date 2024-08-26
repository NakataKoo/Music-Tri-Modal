import librosa
import torch
import numpy as np

audio_path = "/home_lab/nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/audio/lmd_matched_mp3/A/A/L/TRAALAH128E078234A.mp3"
sample_rate = 16000
num_samples = 960000
audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
audio = audio.reshape(1, -1)
x = num_samples - audio.shape[1]
padded_audio = np.pad(audio[0], ((0, x)))
audio = np.array([padded_audio])
print(audio)
print(audio.shape)