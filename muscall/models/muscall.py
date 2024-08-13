import numpy as np
import pickle
import torch
from torch import nn
from transformers import CLIPTextModel
# import torch.nn.functional as F

from muscall.modules.textual_heads import TextTransformer
from muscall.modules.audio_ssl import SimCLRAudio
from muscall.modules.audio_backbones import ModifiedResNet
from muscall.modules.MidiBERT.model import *
from transformers import BertConfig

from datasets import load_dataset
from transformers import ClapModel, ClapProcessor
import laion_clap

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

        self.type_loss = config.loss
        self.temperature = config.temperature

        self.clap = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
        self.clap.load_ckpt('/content/Music-Tri-Modal/music_audioset_epoch_15_esc_90.14.pt')
        
        with open("/content/Music-Tri-Modal/muscall/modules/MidiBERT/CP.pkl", 'rb') as f:
          e2w, w2e = pickle.load(f)

        configuration = BertConfig(max_position_embeddings=512, # max_seq_len
                                position_embedding_type='relative_key_query',
                                hidden_size=768 # args.hs
                           )

        self.midi_head = MidiBert(bertConfig=configuration, e2w=e2w, w2e=w2e)

        for param in self.clap.parameters():
          param.requires_grad = False

        projection_dim = config.projection_dim # 最終的な共通のエンベディングの512次元
        audio_dim = 512 # clap audio hidden size
        text_dim = 512 # clap text hidden size
        midi_dim = 768 # MIDI-BERT hidden size

        self.audio_projection = nn.Linear(audio_dim, projection_dim, bias=False) # audio_dimをprojection_dimへ線形変換
        self.text_projection = nn.Linear(text_dim, projection_dim, bias=False)
        self.midi_projection = nn.Linear(midi_dim, projection_dim, bias=False)

        if self.temperature is None:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_audio(self, audio):
        #audio = audio.reshape(1, -1)
        audio_features = self.clap.get_audio_embedding_from_data(audio, use_tensor=False) # オーディオエンコーダで音声エンベディングを抽出
        audio_features = self.audio_projection(audio_features) # 最終的な共通のエンベディングの次元に変換
        return audio_features

    def encode_text(self, text, text_mask):
        text_features = self.clap.get_text_embedding([text])
        text_features = self.text_projection(text_features)
        return text_features
        
    def encode_midi(self, midi):
        midi_features = self.midibert.forward(midi)
        # [ここにベクトル平均化の処理]
        midi_features = self.midi_projection(midi_features)

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

        #loss = (F.cross_entropy(logits_per_pc_text, self.labels) + \
        #        F.cross_entropy(logits_per_text_pc, self.labels)) / 2 + \
        #        (F.cross_entropy(logits_per_pc_image, self.labels) + F.cross_entropy(logits_per_image_pc, self.labels)) / 2

        # マルチモーダル損失を計算
        if return_loss:
            loss = clip_loss(logits_per_text_midi) + clip_loss(logits_per_audio_midi)
            return loss
        #else:
        #    return logits_per_audio, logits_per_text

    @classmethod
    def config_path(cls):
        return "configs/models/muscall.yaml"
