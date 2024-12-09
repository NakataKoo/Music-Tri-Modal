import os
import time
import numpy as np
import miditoolkit
import torch
import glob
import tqdm
import random

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from pytorch_memlab import profile
from pytorch_memlab import MemReporter

from muscall.datasets.emopia import EMOPIA
from muscall.datasets.pianist8 import Pianist8
from muscall.datasets.vgmidi import VGMIDI
from muscall.datasets.wikimt import WIKIMT

from muscall.trainers.base_trainer import BaseTrainer
from muscall.models.muscall import MusCALL
from muscall.utils.audio_utils import get_transform_chain

from muscall.models.finetune_muscall import SequenceClassification

# parserなどで指定
seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)

class MusCALLFinetuner(BaseTrainer):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.batch_size = self.config.training.dataloader.batch_size
        self.config = config
        self.load() # load_dataset()、build_model()、build_optimizer()、self.logger.save_config()の実行
        self.scaler = torch.amp.GradScaler()

    def load_dataset(self):
        self.logger.write("Loading dataset")
        self.dataset_name = self.config.dataset_config.dataset_name

        # AudioCaptionMidiDatasetのインスタンスを生成（audiocaption.yamlの内容を引数に指定）
        if self.dataset_name == "piansit8":
            self.train_dataset = Pianist8(self.config.dataset_config, dataset_type="train", midi_dic=self.config.model_config.midi.midi_dic)
            self.val_dataset = Pianist8(self.config.dataset_config, dataset_type="val", midi_dic=self.config.model_config.midi.midi_dic)
        elif self.dataset_name == "emopia":
            self.train_dataset = EMOPIA(self.config.dataset_config, dataset_type="train", midi_dic=self.config.model_config.midi.midi_dic)
            self.val_dataset = EMOPIA(self.config.dataset_config, dataset_type="val", midi_dic=self.config.model_config.midi.midi_dic)
        elif self.dataset_name == "vgmidi":
            self.train_dataset = VGMIDI(self.config.dataset_config, dataset_type="train", midi_dic=self.config.model_config.midi.midi_dic)
            self.val_dataset = VGMIDI(self.config.dataset_config, dataset_type="val", midi_dic=self.config.model_config.midi.midi_dic)
        elif self.dataset_name == "wikimt":
            self.train_dataset = WIKIMT(self.config.dataset_config, dataset_type="train", midi_dic=self.config.model_config.midi.midi_dic)
            self.val_dataset = WIKIMT(self.config.dataset_config, dataset_type="val", midi_dic=self.config.model_config.midi.midi_dic)
        else:
            raise ValueError("{} dataset is not supported.".format(dataset_name))

        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            **self.config.training.dataloader,
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        self.val_loader = DataLoader(
            dataset=self.val_dataset,
            **self.config.training.dataloader,
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

        self.logger.write(
            "Number of training samples: {}".format(self.train_dataset.__len__())
        ) # サンプル数をログに出力

    def build_model(self):
        self.logger.write("Building model")
        self.midibert = MusCALL(self.config.model_config, is_train=True).to(self.device)

        if torch.cuda.device_count() > 1:
            print("Use %d GPUS" % torch.cuda.device_count())
            self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1])

        self.model = SequenceClassification(self.midibert, 
                                            class_num=self.train_dataset.num_classes(), 
                                            hs=self.config.model_config.projection_dim).to(self.device)

        self.reporter = MemReporter(self.model)
        # self.reporter.report()

    def build_optimizer(self):
        self.logger.write("Building optimizer")
        optimizer_config = self.config.training.optimizer
        self.optimizer = getattr(optim, optimizer_config.name, None)(
            self.model.parameters(), **optimizer_config.args
        )

        num_train_optimization_steps = (
            int(self.train_loader.dataset.__len__() / self.batch_size)
            * self.config.training.epochs
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=num_train_optimization_steps * 0.1
        )

    # train.pyによって実行されるメソッド
    def train(self):
        best_r10 = 0 # 最良のR@10スコアを追跡

        if os.path.exists(self.logger.checkpoint_path):
            self.logger.write(
                "Resumed training experiment with id {}".format(
                    self.logger.experiment_id
                )
            )
            self.load_ckpt(self.logger.checkpoint_path) # チェックポイントからの再開
        else: # 学習の新規開始
            self.logger.write(
                "Started training experiment with id {}".format(
                    self.logger.experiment_id
                )
            )
            self.start_epoch = 0

        torch.backends.cudnn.benchmark = True

        for epoch in range(self.start_epoch, self.config.training.epochs): # start_epoch=0 ~ max epochs
            epoch_start_time = time.time()

            train_loss = self.train_epoch(self.train_loader, is_training=True)
            val_loss = self.train_epoch_val(self.val_loader)

            # バッチごとの進捗を表示
            print(f"Epoch [{epoch + 1}/{self.config.training.epochs}], train loss: {train_loss}, val loss: {val_loss}")

            torch.cuda.empty_cache()

            r10 = 0
            if self.config.training.track_retrieval_metrics:
                r10 = self.get_retrieval_metrics() # 検索メトリクスの取得

            epoch_time = time.time() - epoch_start_time
            self.logger.update_training_log(
                epoch + 1,
                train_loss,
                val_loss,
                epoch_time,
                self.scheduler.get_last_lr()[0],
                r10,
            )

            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }

            is_best = r10 > best_r10
            if is_best:
                best_r10 = r10
            # save checkpoint in appropriate path (new or best) (最良のモデルと最新のモデルを保存)
            self.logger.save_checkpoint(state=checkpoint, is_best=is_best)

            # self.reporter.report()

    def load_ckpt(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.start_epoch = checkpoint["epoch"]

    # 1エポック分の学習を実行（各バッチの損失計算＋バックプロパゲーション）
    def train_epoch(self, data_loader, is_training):
        running_loss = 0.0
        n_batches = 0

        if is_training:
            self.model.train()
        else:
            self.model.eval()

        pbar = tqdm.tqdm(data_loader, disable=False, leave=True)

        # データローダーからバッチを取り出し、学習を実行
        for i, batch in enumerate(pbar):
            batch = tuple(t.to(device=self.device, non_blocking=True) if isinstance(t, torch.Tensor) else t for t in batch)

            if self.dataset_name == "wikimt":
                labels, _ , input_midi, first_input_midi_shape, _ = batch
            else:
                labels, input_midi, first_input_midi_shape, _ = batch




            with torch.amp.autocast("cuda", enabled=self.config.training.amp):
                loss = self.model(
                    input_audio,
                    input_text,
                    input_midi,
                    first_input_midi_shape,
                    original_audio=original_audio, # 元の音声データ（拡張前の音声データ）
                    sentence_sim=sentence_sim,# 文の類似度（オプション:損失関数がweighted_clipの場合）
                )
            
            if not self.config.training.amp:
                loss = self.model(
                        input_audio,
                        input_text,
                        input_midi,
                        first_input_midi_shape,
                        original_audio=original_audio, # 元の音声データ（拡張前の音声データ）
                        sentence_sim=sentence_sim,# 文の類似度（オプション：損失関数がweighted_clipの場合）
                )

            # 逆誤差伝播とパラメータ更新
            if is_training:
                if self.config.training.amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                # clamp temperature scaling if over log(100)
                if self.model.logit_scale.item() > np.log(100):
                    self.model.logit_scale.data = torch.clamp(
                        self.model.logit_scale.data, max=np.log(100)
                    )

                self.scheduler.step()
                self.optimizer.zero_grad()

            running_loss += loss.item() # 各バッチの損失を蓄積
            del loss
            n_batches += 1 # バッチ数をカウント

        return running_loss / n_batches

    def train_epoch_val(self, data_loader):
        with torch.no_grad():
            loss = self.train_epoch(data_loader, is_training=False)
        return loss
