import os
import time
import numpy as np
import miditoolkit
import torch
import pytorch_lightning as pl
import glob

from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR

from muscall.datasets.audiocaption import AudioCaptionMidiDataset
from muscall.models.muscall import MusCALL
from muscall.tasks.retrieval import run_retrieval
from muscall.utils.audio_utils import get_transform_chain


class MusCALLLightningModule(pl.LightningModule):
    def __init__(self, config):
        super(MusCALLLightningModule, self).__init__()
        self.config = config
        self.model = self.build_model()
        self.scaler = torch.cuda.amp.GradScaler()

    def build_model(self):
        model_name = self.config.model_config.model_name
        if model_name == "muscall":
            return MusCALL(self.config.model_config)
        else:
            raise ValueError(f"{model_name} model is not supported.")
        
    def forward(self, input_audio, input_text, input_midi, first_input_midi_shape, original_audio=None, sentence_sim=None):
        return self.model(input_audio, input_text, input_midi, first_input_midi_shape, original_audio, sentence_sim)

    def configure_optimizers(self):
        optimizer_config = self.config.training.optimizer
        optimizer = getattr(torch.optim, optimizer_config.name)(self.model.parameters(), **optimizer_config.args)

        num_train_optimization_steps = (
            int(self.config.training.dataloader.dataset_len / self.config.training.dataloader.batch_size)
            * self.config.training.epochs
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=num_train_optimization_steps * 0.1)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        audio_id, input_audio, input_text, input_midi, first_input_midi_shape, midi_dir_paths, idx = batch
        
        # Data augmentation if enabled
        audio_data_config = self.config.dataset_config.audio
        original_audio = None
        if self.training and audio_data_config.augment:
            original_audio = input_audio
            augment_chain = get_transform_chain(
                p_polarity=0, 
                p_gain=0,
                p_noise=audio_data_config.p_noise,
                p_pitch_shift=audio_data_config.p_pitch_shift,
                sample_rate=audio_data_config.sr
            )
            input_audio = augment_chain(input_audio.unsqueeze(1), audio_data_config.sr).squeeze(1)

        loss = self(input_audio, input_text, input_midi, first_input_midi_shape, original_audio)

        return loss

    def validation_step(self, batch, batch_idx):
        audio_id, input_audio, input_text, input_midi, first_input_midi_shape, midi_dir_paths, idx = batch
        loss = self(input_audio, input_text, input_midi, first_input_midi_shape)

        return loss

    def get_retrieval_metrics(self):
        indices = torch.randperm(len(self.val_dataset))[:1000]
        random_val_subset = Subset(self.val_dataset, indices)
        val_subset_loader = DataLoader(
            random_val_subset,
            batch_size=self.config.training.dataloader.batch_size,
        )
        retrieval_metrics_midi_audio = run_retrieval(
            model=self.model,
            data_loader=val_subset_loader,
            device=self.device,
            retrieval_type="midi_audio"
        )

        retrieval_metrics_midi_text = run_retrieval(
            model=self.model,
            data_loader=val_subset_loader,
            device=self.device,
            retrieval_type="midi_text"
        )

        r10 = (retrieval_metrics_midi_audio["R@10"].item() + retrieval_metrics_midi_text["R@10"].item()) / 2

        return r10


class MusCALLDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super(MusCALLDataModule, self).__init__()
        self.config = config

    def setup(self, stage=None):
        dataset_name = self.config.dataset_config.dataset_name
        if dataset_name == "audiocaptionmidi":
            self.train_dataset = AudioCaptionMidiDataset(self.config.dataset_config, dataset_type="train", midi_dic=self.config.model_config.midi.midi_dic)
            self.val_dataset = AudioCaptionMidiDataset(self.config.dataset_config, dataset_type="val", midi_dic=self.config.model_config.midi.midi_dic)
        else:
            raise ValueError(f"{dataset_name} dataset is not supported.")
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.training.dataloader.batch_size, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.training.dataloader.batch_size, drop_last=True)
