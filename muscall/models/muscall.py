import numpy as np

import torch
from torch import nn
from transformers import CLIPTextModel

from muscall.modules.textual_heads import TextTransformer
from muscall.modules.audio_ssl import SimCLRAudio
from muscall.modules.audio_backbones import ModifiedResNet

# クロスエントロピー誤差
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    labels = torch.arange(len(logits), device=logits.device) # 
    return nn.functional.cross_entropy(logits, labels)

# 誤差関数
def clip_loss(similarity: torch.Tensor, sentence_sim=None, type_loss="clip") -> torch.Tensor:
    loss1 = contrastive_loss(similarity)
    loss2 = contrastive_loss(similarity.T)
    return (loss1 + loss2) / 2.0


class MusCALL(nn.Module):
    def __init__(self, config):
        super().__init__()
        audio_config = config.audio # 音声エンコーダの設定
        text_config = config.text # テキストエンコーダの設定
        midi_config = config.midi # midiテキストエンコーダの設定

        projection_dim = config.projection_dim
        audio_dim = audio_config.hidden_size
        text_dim = text_config.hidden_size
        midi_dim = midi_config.hidden_size
        
        self.type_loss = config.loss
        self.temperature = config.temperature

        if config.audio.model == "ModifiedResNet":
            self.audio_backbone = ModifiedResNet(audio_config)
        if config.text.model == "TextTransformer":
            self.textual_head = TextTransformer(text_config)
        elif config.text.model == "CLIPTextModel":
            pretrained_model = config.text.pretrained
            self.textual_head = CLIPTextModel.from_pretrained(pretrained_model)
        elif config.text.model == "CLAPTextModel":
		        #[定義する必要あり]
		        #self.textual_head = ○○
        if config.midi.model == "MusicBERT":
		        #[定義する必要あり]
		        #self.midi_head = ○○
		    
        # テキストエンコーダのパラメータを凍結
        for param in self.textual_head.parameters():
            param.requires_grad = False
        # 音声エンコーダのパラメータを凍結
        for param in self.audio_backbone.parameters():
		        param.requires_grad = False

        self.audio_projection = nn.Linear(audio_dim, projection_dim, bias=False)
        self.text_projection = nn.Linear(text_dim, projection_dim, bias=False)
        self.midi_projection = nn.Linear(midi_dim, projection_dim, bias=False)

        if self.temperature is None:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_audio(self, audio):
        audio_features = self.audio_backbone(audio)
        audio_features = self.audio_projection(audio_features)
        return audio_features

    def encode_text(self, text, text_mask):
        if isinstance(self.textual_head, TextTransformer):
            text_features = self.textual_head(text, text_mask)
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            pooled_outout = text_features[
                torch.arange(text_features.shape[0]), text.argmax(dim=-1)
            ]
        elif isinstance(self.textual_head, CLIPTextModel):
            outputs = self.textual_head(text, text_mask)
            pooled_outout = outputs.pooler_output

        text_features = self.text_projection(pooled_outout)
        return text_features
        
    def encode_midi(self, midi):
	    [定義する必要あり]

    # 音声とテキストの特徴をエンコードし、対照学習のための損失を計算
    def forward(
        self,
        audio, # 拡張後の音声データ（バッチ）
        text, # テキストデータ（バッチ）
        midi, 
        original_audio=None, # 元音声データ
        sentence_sim=None, # 文の類似度(オプション)
        text_mask=None, #テキストのマスク(オプション)
        return_loss=True, # 損失を計算して返すかどうかを指定するフラグ
    ):

        # 音声とテキストとmidiの特徴をそれぞれエンコード
        audio_features = self.encode_audio(audio)
        text_features = self.encode_text(text, text_mask)
        midi_fieatures = self.encode_midi(midi)

        # normalise features（各特徴ベクトルをそのノルムで割ることで、単位ベクトルに変換）
        audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        midi_features = midi_features / midi_features.norm(dim=-1, keepdim=True)

        # ロジットの計算
        # ロジットスケール（温度パラメータ）を計算。温度が未設定の場合、学習されたlogit_scaleを使用
        if self.temperature is None:
            logit_scale = self.logit_scale.exp()
        # 音声とテキストの特徴ベクトルの内積を計算してロジットを得る
        else:
            logit_scale = 1.0 / self.temperature

        logits_per_midi_text = logit_scale * midi_features @ text_features.t()
        logits_per_text_midi = logits_per_midi_text.t()
        logits_per_midi_audio = logit_scale * midi_features @ audio_features.t()
        logits_per_audio_midi = logits_per_midi_audio.t()

        loss = (F.cross_entropy(logits_per_pc_text, self.labels) + \
                F.cross_entropy(logits_per_text_pc, self.labels)) / 2 + \
                (F.cross_entropy(logits_per_pc_image, self.labels) + F.cross_entropy(logits_per_image_pc, self.labels)) / 2


        # マルチモーダル損失を計算
        if return_loss:
            loss = clip_loss(logits_per_text_midi) + clip_loss(logits_per_audio_midi)
            return loss
        else:
            return logits_per_audio, logits_per_text

    @classmethod
    def config_path(cls):
        return "configs/models/muscall.yaml"