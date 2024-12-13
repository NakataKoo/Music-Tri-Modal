import os
import time
import numpy as np
import miditoolkit
import torch
import glob
import tqdm
import random
import pandas as pd
import argparse

from muscall.models.muscall import MusCALL
from muscall.utils.audio_utils import get_transform_chain
from muscall.utils.utils import (
    load_conf,
    merge_conf,
    get_root_dir,
    update_conf_with_cli_params,
)

seed = 0
np.random.seed(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a MusCALL model")

    parser.add_argument("--experiment_id",type=str,default=None)
    parser.add_argument("--checkpoint",type=str,default=None)
    parser.add_argument("--finetune_config",type=str,default=os.path.join(get_root_dir(), "configs", "finetuning.yaml"))
    parser.add_argument("--model_config",type=str,default=os.path.join(get_root_dir(), "configs", "models", "muscall.yaml"))
    parser.add_argument("--dataset", type=str, help="name of the dataset", default="pianist8")
    parser.add_argument("--midi_size", type=int, help="name of the dataset", default=10)

    args = parser.parse_args()
    return args

def midi_padding(input_midi, midi_size):

    input_midi = torch.tensor(input_midi)

    # パディング処理
    if input_midi.size(0) < midi_size:
        x = midi_size - input_midi.size(0)
        input_midi = torch.nn.functional.pad(input_midi, (0, 0, 0, 0, 0, x))

      # クロップ処理
    elif input_midi.size(0) > midi_size:
        input_midi = input_midi[:midi_size, :]

    return input_midi


if __name__ == "__main__":
    params = parse_args()
        
    model_config = load_conf(params.model_config)
    finetune_config = load_conf(params.finetune_config)

    if params.dataset == 'pianist8':
      num_classes = 8
    elif params.dataset == 'emopia':
      num_classes = 4

    checkpoint_path = params.checkpoint
    feature_extractor = MusCALL(model_config.model_config, is_train=False).to("cuda")
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    feature_extractor.load_state_dict(checkpoint["state_dict"])
    feature_extractor.eval()

    data_root = os.path.join(finetune_config.env.data_root, "datasets", params.dataset)

    X = np.load(os.path.join(data_root, f'{params.dataset}.npy'), allow_pickle=True)
    y = np.load(os.path.join(data_root, f'{params.dataset}_ans.npy'), allow_pickle=True)

    data_list = []
    for feature, label in zip(X, y):
          feature_pad = midi_padding(feature, params.midi_size)
          feature_new = feature_extractor.encode_midi_per_data(feature_pad).cpu().detach().numpy()
            
          # 多次元配列をflattenして1次元に
          flattened = feature_new.flatten()

          # 要素数に合わせて x1, x2, ... という列名を作成
          col_names = [f'x{i+1}' for i in range(len(flattened))]

          # flattened配列から1行分のデータを辞書化（ラベルも含む）
          row_dict = {name: val for name, val in zip(col_names, flattened)}
          row_dict['label'] = label

          data_list.append(row_dict)

    # 全ての行をDataFrame化
    df = pd.DataFrame(data_list)
    df.to_csv(params.dataset+".csv", index=False)