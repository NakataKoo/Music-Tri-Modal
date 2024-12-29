import numpy as np
import pickle
import torch
from torch import nn
from pytorch_memlab import profile

from muscall.modules.MidiBERT.model import *
from transformers import BertConfig, AlbertConfig, DistilBertConfig #RobertaConfig,

# クロスエントロピー誤差
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    labels = torch.arange(len(logits), device=logits.device)
    return nn.functional.cross_entropy(logits, labels)

# 誤差関数
def clip_loss(similarity: torch.Tensor, type_loss="clip") -> torch.Tensor:
    loss1 = contrastive_loss(similarity)
    loss2 = contrastive_loss(similarity.T)
    return (loss1 + loss2) / 2.0

class MusCALL(nn.Module):
    def __init__(self, config, is_train=True):
        super().__init__()

        self.type_loss = config.loss
        self.temperature = config.temperature
        self.device = torch.device("cuda")
        
        with open(config.midi.midi_dic, 'rb') as f:
            e2w, w2e = pickle.load(f)

        if config.midi.model_name == 'bert':
            print("Build BERT\n")
            configuration = BertConfig(max_position_embeddings=512,
                                    position_embedding_type='relative_key_query',
                                    hidden_size=768,
                                    num_hidden_layers = 6,
                                    attn_implementation="eager",
                                    vocab_size = 800,
                                    hidden_dropout_prob=0.3,
                                    attention_probs_dropout_prob=0.3,
            )
            self.midi_dim = configuration.hidden_size
        elif config.midi.model_name == 'albert':
            print("Build ALBERT\n")
            configuration = AlbertConfig(attention_probs_dropout_prob=0.1,
                                    hidden_dropout_prob=0.1,
                                    embedding_size=128,
                                    hidden_size=768,
                                    initializer_range=0.02,
                                    intermediate_size=3072,
                                    max_position_embeddings=512,
                                    num_attention_heads=12,
                                    num_hidden_layers=12,
                                    num_hidden_groups=1,
                                    net_structure_type=0,
                                    gap_size=0,
                                    num_memory_blocks=0,
                                    inner_group_num=1,
                                    down_scale_factor=1,
                                    type_vocab_size=2,
                                    vocab_size=800,
                                    position_embedding_type='relative_key_query',
                                    attn_implementation="eager"
            )
            self.midi_dim = configuration.hidden_size
        elif config.midi.model_name == 'distilbert':
            print("Build DistilBERT\n")
            configuration = DistilBertConfig(max_position_embeddings=512,
                                    dim = 768,
                                    vocab_size = 800
            )
            self.midi_dim = configuration.dim
        self.midibert = MidiBert(bertConfig=configuration, e2w=e2w, w2e=w2e, model_name=config.midi.model_name)

        stdict_o = None
        print("Load Checkpoint?: "+str(config.midi.load_ckpt))
        if config.midi.load_ckpt and is_train:
            print("\nLoad Check point to restart\n")
            print(config.midi.ckpt)
            cpt = torch.load(config.midi.ckpt, weights_only=False)
            stdict_m = cpt['state_dict']
            stdict_o = cpt['optimizer']
            self.midibert.load_state_dict(stdict_m, strict=False)

        self.projection_dim = config.projection_dim # 最終的な共通のエンベディングの512次元
        self.midi_projection = nn.Linear(self.midi_dim, self.projection_dim, bias=False)

        if self.temperature is None:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def encode_midi(self, midi_batch, first_input_midi_shape=None):
        '''
        midi_batch: torch.Size([batch_size, midi_size, 512, 4])
        first_input_midi_shape: torch.Size([batch_size]) (バッチ内の各データの初期midi数が記載)
        midiを個別に処理したい
        '''
        embedding_midi = []
        for midi in midi_batch:

            midi_features = self.midibert.forward(midi) # 最もGPUを消費
            midi_features_all = midi_features.last_hidden_state.mean(dim=1).mean(dim=0)
            
            # 同一のmidiのfeaturesで平均化
            midi_features_all = self.midi_projection(midi_features_all)
            embedding_midi.append(midi_features_all)
            
        embedding_midi = torch.stack(embedding_midi, dim=0) # torch.Tensorが入ったlistを二次元のTensorに変換
        return embedding_midi
    
    def encode_midi_per_data(self, midi):
        # midi: torch.Size([midi_size, 512, 4])
        midi_features = self.midibert.forward(midi)
        midi_features_all = midi_features.last_hidden_state.mean(dim=1).mean(dim=0)
        midi_features_all = self.midi_projection(midi_features_all)
        return midi_features_all

    # 音声とテキストの特徴をエンコードし、対照学習のための損失を計算
    def forward(
        self,
        text_features, # テキストデータ（バッチ）
        midi,  # midiデータ（バッチ）
        first_input_midi_shape,
        return_loss=True, # 損失を計算して返すかどうかを指定するフラグ
    ):

        # midiの特徴をそれぞれエンコード
        midi_features = self.encode_midi(midi, first_input_midi_shape)

        # normalise features（各特徴ベクトルをそのノルムで割ることで、単位ベクトルに変換）
        epsilon = 1e-6
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + epsilon)
        midi_features = midi_features / (midi_features.norm(dim=-1, keepdim=True) + epsilon)

        # ロジットの計算
        # ロジットスケール（温度パラメータ）を計算。温度が未設定の場合、学習されたlogit_scaleを使用
        if self.temperature is None:
            logit_scale = self.logit_scale.exp()
        # 音声とテキストの特徴ベクトルの内積を計算してロジットを得る
        else:
            logit_scale = 1.0 / self.temperature

        logits_per_midi_text = logit_scale * midi_features @ text_features.t()
        logits_per_text_midi = logits_per_midi_text.t()

        # マルチモーダル損失を計算
        if return_loss:
            loss = clip_loss(logits_per_text_midi) #+ clip_loss(logits_per_audio_midi)
            print(f"loss: {loss}")
            return loss

    @classmethod
    def config_path(cls):
        return "configs/models/muscall.yaml"
