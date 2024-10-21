import argparse
import os
from omegaconf import OmegaConf
import pytorch_lightning as pl
from datetime import datetime

from muscall.utils.logger import Logger
from muscall.utils.utils import (
    load_conf,
    merge_conf,
    get_root_dir,
    update_conf_with_cli_params,
)
from muscall.models.muscall import MusCALL
from muscall.trainers.muscall_trainer_pl import MusCALLLightningModule, MusCALLDataModule
from muscall.datasets.audiocaption import AudioCaptionMidiDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train a MusCALL model")

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
        default=os.path.join(get_root_dir(), "configs", "training.yaml"),  # デフォルトではtraining.yamlを読み込む
    )
    parser.add_argument(
        "--dataset", type=str, help="name of the dataset", default="audiocaption"
    )
    parser.add_argument("--device_num", type=str, default="0, 1")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    params = parse_args()

    if params.experiment_id is None:
        # 1. Load config (base + dataset + model)
        base_conf = load_conf(params.config_path)  # デフォルトではtraining.yamlを読み込む

        if params.dataset == "audiocaption":
            dataset_conf_path = os.path.join(base_conf.env.base_dir, AudioCaptionMidiDataset.config_path())  # "configs/datasets/audiocaption.yaml"のパスを取得
        else:
            raise ValueError(f"{params.dataset} dataset not supported")

        model_conf_path = os.path.join(base_conf.env.base_dir, MusCALL.config_path())  # "configs/models/muscall.yaml"のパスを取得

        config = merge_conf(params.config_path, dataset_conf_path, model_conf_path)  # 「config＝training設定＋データセット設定＋モデル設定」(config.yamlが学習設定ファイルとして保存される)
        update_conf_with_cli_params(params, config)  # 引数の値をconfigに反映
    else:
        config = OmegaConf.load(f"./save/experiments/{params.experiment_id}/config.yaml")

    logger = Logger(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = params.device_num

    checkpoint_dir = "save/experiments/"+datetime.now().strftime("%Y%m%d")

    # Pytorch Lightningのトレーナー設定
    trainer = pl.Trainer(
        max_epochs=config.training.epochs,
        accelerator="gpu", 
        devices=2, 
        #strategy="ddp",
        default_root_dir=checkpoint_dir,  # ログとチェックポイントを保存するディレクトリ
        # precision=16 if config.training.amp else 32,  # AMP（混合精度）を有効化
        callbacks=[pl.callbacks.ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1)]  # バリデーション損失に基づくチェックポイントを保存
    )

    # Lightning Module（モデルとトレーニングロジック）とデータモジュールの初期化
    model = MusCALLLightningModule(config)
    data_module = MusCALLDataModule(config)

    # 学習開始
    trainer.fit(model, datamodule=data_module)