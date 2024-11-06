import torch
import json
from tqdm import tqdm
import glob
from muscall.datasets.model import CP
import numpy as np

data = glob.glob('/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/midi/audio2midi/**/*.mid', recursive=True)

pbar = tqdm(data, disable=False, leave=True)
cp = CP(dict="/home/Nakata/Music-Tri-Modal/muscall/modules/MidiBERT/CP.pkl")

files = []

for obj in pbar:
  all_words = cp.prepare_data(midi_paths=[obj], task="", max_len=512) 
  npy_file_path = obj.replace('.mid', '.npy')
  np.save(npy_file_path, all_words)
