import glob
import torch
import os
from muscall.datasets.model import CP

files = glob.glob('/home/Nakata/Music-Tri-Modal/data/datasets/audiocaptionmidi/basic-pitch-output/*.mid', recursive=True)
cp = CP(dict="/home/Nakata/Music-Tri-Modal/muscall/modules/MidiBERT/CP.pkl")

ok = []
del_data = []
for file in files:
    all_words = cp.prepare_data(midi_paths=[file], task="", max_len=512) 
    if all_words.shape == torch.Size([0]):
        del_data.append(file)
    else:
        ok.append(file)

for file in del_data:
    print("del: "+file)
    #os.remove(file)