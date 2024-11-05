import numpy as np
import pickle
import torch
from torch import nn
from pytorch_memlab import profile

from muscall.modules.MidiBERT.model import *
from transformers import BertConfig, AlbertConfig, RobertaConfig, DistilBertConfig
import laion_clap

# クロスエントロピー誤差
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    labels = torch.arange(len(logits), device=logits.device)
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
        self.device = torch.device("cuda")

        self.clap = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base', device=self.device)
        self.clap.load_ckpt(config.clap.clap_ckpt, verbose=False)
        
        with open(config.midi.midi_dic, 'rb') as f:
            e2w, w2e = pickle.load(f)

        if config.midi.model_name == 'bert':
            print("Build BERT\n")
            configuration = BertConfig(max_position_embeddings=512,
                                    position_embedding_type='relative_key_query',
                                    hidden_size=768,
                                    num_hidden_layers = 6,
                                    attn_implementation="eager",
                                    vocab_size = 800
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
        if config.midi.load_ckpt:
            print("\nLoad Check point to restart\n")
            print(config.midi.ckpt)
            cpt = torch.load(config.midi.ckpt, weights_only=True)
            stdict_m = cpt['state_dict']
            stdict_o = cpt['optimizer']
            self.midibert.load_state_dict(stdict_m, strict=False)

        for param in self.clap.parameters():
            param.requires_grad = False

        self.projection_dim = config.projection_dim # 最終的な共通のエンベディングの512次元
        self.audio_dim = config.clap.audio_hidden_size
        self.text_dim = config.clap.text_hidden_size

        self.audio_projection = nn.Linear(self.audio_dim, self.projection_dim, bias=False) # audio_dimをprojection_dimへ線形変換
        self.text_projection = nn.Linear(self.text_dim, self.projection_dim, bias=False)
        self.midi_projection = nn.Linear(self.midi_dim, self.projection_dim, bias=False)

        if self.temperature is None:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    @torch.no_grad()
    def encode_audio(self, audio):
        """
        audio: batchで来る, torch.Tensor型
        """

        audio_features = []
        for data in audio:
            
            data = torch.nan_to_num(data, nan=0.0) # nanを0に変換
            audio_feature = self.clap.get_audio_embedding_from_data(data, use_tensor=True) # 音声エンベディングを抽出
            
            # numpy.ndarray から torch.Tensor に変換
            if isinstance(audio_feature, np.ndarray):
                audio_feature = torch.from_numpy(audio_feature.astype(np.float32)).clone()
            
            # デバイスに移動
            audio_feature = audio_feature.to(self.device)
            audio_feature = self.audio_projection(audio_feature) # 最終的な共通のエンベディングの次元に変換
            audio_features.append(audio_feature)
        audio_features = torch.cat(audio_features, dim=0)
        return audio_features

    @torch.no_grad()
    def encode_text(self, text, text_mask):
        # もしタプルだったら、リストに変換
        if isinstance(text, tuple):
            text = list(text)
        # textがリストではない場合、リストに変換
        if not isinstance(text, list):
            text = [text]
        text_features = self.clap.get_text_embedding(text, use_tensor=True)
    
        # デバイスに移動
        text_features = text_features.to(self.device)
        text_features = self.text_projection(text_features)
        return text_features
    
    def encode_midi(self, midi_batch, first_input_midi_shape):
        '''
        midi_batch: torch.Size([batch_size, midi_size, 512, 4])
        first_input_midi_shape: torch.Size([batch_size]) (バッチ内の各データの初期midi数が記載)
        midiを個別に処理したい
        '''
        embedding_midi = []
        for midi, midi_shape in zip(midi_batch, first_input_midi_shape):

            # 元のmidiサイズに戻す
            midi = midi[0:midi_shape]

            midi_features = self.midibert.forward(midi) # 最もGPUを消費
            midi_features_all = torch.zeros(self.midi_dim, device=self.device)  # デバイス上で初期化
            # トークンのベクトルを平均して、シーケンス全体のベクトルを生成
            for i in range(len(midi)):
                midi_features_all = midi_features_all + midi_features.last_hidden_state[i].mean(dim=0) # (batch_size, hidden_size)
            midi_features_all = midi_features_all / len(midi)
            
            # 同一のmidiのfeaturesで平均化
            midi_features_all = self.midi_projection(midi_features_all)
            embedding_midi.append(midi_features_all)
            
            del midi_features_all
            del midi

        embedding_midi = torch.stack(embedding_midi, dim=0) # torch.Tensorが入ったlistを二次元のTensorに変換
        return embedding_midi

    # 音声とテキストの特徴をエンコードし、対照学習のための損失を計算
    def forward(
        self,
        audio, # 拡張後の音声データ（バッチ）
        text, # テキストデータ（バッチ）
        midi,  # midiデータ（バッチ）
        first_input_midi_shape,
        original_audio=None, # 元音声データ
        sentence_sim=None, # 文の類似度(オプション)
        text_mask=None, #テキストのマスク(オプション)
        return_loss=True, # 損失を計算して返すかどうかを指定するフラグ
    ):

        # 音声とテキストとmidiの特徴をそれぞれエンコード
        audio_features = self.encode_audio(audio)
        text_features = self.encode_text(text, text_mask)
        midi_features = self.encode_midi(midi, first_input_midi_shape)

        # normalise features（各特徴ベクトルをそのノルムで割ることで、単位ベクトルに変換）
        epsilon = 1e-6
        audio_features = audio_features / (audio_features.norm(dim=-1, keepdim=True) + epsilon)
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
        logits_per_midi_audio = logit_scale * midi_features @ audio_features.t()
        logits_per_audio_midi = logits_per_midi_audio.t()

        # マルチモーダル損失を計算
        if return_loss:
            loss = clip_loss(logits_per_text_midi) + clip_loss(logits_per_audio_midi)
            print(f"loss: {loss}")
            return loss

    @classmethod
    def config_path(cls):
        return "configs/models/muscall.yaml"
