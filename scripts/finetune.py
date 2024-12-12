import argparse
import os
from omegaconf import OmegaConf

from muscall.utils.logger import Logger
from muscall.utils.utils import (
    load_conf,
    merge_conf,
    get_root_dir,
    update_conf_with_cli_params,
)

from muscall.models.muscall import MusCALL
from muscall.trainers.muscall_finetuner import MusCALLFinetuner

from muscall.datasets.emopia import EMOPIA
from muscall.datasets.pianist8 import Pianist8
from muscall.datasets.vgmidi import VGMIDI
from muscall.datasets.wikimt import WIKIMT

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a MusCALL model")

    parser.add_argument(
        "--experiment_id",
        type=str,
        help="experiment id under which checkpoint was saved",
        default=None,
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="path to base config file",
        default=os.path.join(get_root_dir(), "configs", "finetuning.yaml"),
    )
    parser.add_argument(
        "--dataset", type=str, help="name of the dataset", default="piansit8"
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    params = parse_args()
        
    base_conf = load_conf(params.config_path)

    model_conf_path = OmegaConf.load(
            "./save/experiments/{}/config.yaml".format(params.experiment_id)
    )

    model_conf_path = model_conf_path.model_config

    config = merge_conf(params.config_path, model_conf_path)
    update_conf_with_cli_params(params, config)

    logger = Logger(config)

    finetuner = MusCALLFinetuner(config, logger, params.dataset)
    # print("# of trainable parameters:", finetuner.count_parameters()) # 学習パラメータ数の表示

    finetuner.finetune() # モデル学習開始
