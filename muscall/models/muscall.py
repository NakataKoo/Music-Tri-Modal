import numpy as np
import pickle
import torch
from torch import nn

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

        self.clap = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
        self.clap.load_ckpt(config.clap.clap_ckpt)
        
        with open(config.midi.midi_dic, 'rb') as f:
            e2w, w2e = pickle.load(f)

        if config.midi.model_name == 'bert':
            configuration = BertConfig(max_position_embeddings=config.midi.max_seq_len, # 512
                                    position_embedding_type='relative_key_query',
                                    hidden_size=config.midi.hidden_size, # 768
                                    num_attention_heads = config.midi.num_attention_heads,
                                    num_hidden_layers = config.midi.num_hidden_layers,
                                    intermediate_size = config.midi.intermediate_size,
                                    vocab_size = config.midi.vocab_size
            )
        elif config.midi.model_name == 'albert':
            configuration = AlbertConfig(max_position_embeddings=config.midi.max_seq_len, # 512
                                    position_embedding_type='relative_key_query',
                                    hidden_size=config.midi.hidden_size, # 768
                                    num_attention_heads = config.midi.num_attention_heads,
                                    num_hidden_layers = config.midi.num_hidden_layers,
                                    intermediate_size = config.midi.intermediate_size,
                                    vocab_size = config.midi.vocab_size
            )
        elif config.midi.model_name == 'roberta':
            configuration = RobertaConfig(max_position_embeddings=config.midi.max_seq_len, # 512
                                    position_embedding_type='relative_key_query',
                                    hidden_size=config.midi.hidden_size, # 768
                                    num_attention_heads = config.midi.num_attention_heads,
                                    num_hidden_layers = config.midi.num_hidden_layers,
                                    intermediate_size = config.midi.intermediate_size,
                                    vocab_size = config.midi.vocab_size
            )
        elif config.midi.model_name == 'distilbert':
            configuration = DistilBertConfig(max_position_embeddings=config.midi.max_seq_len, # 512
                                    position_embedding_type='relative_key_query',
                                    hidden_size=config.midi.hidden_size, # 768
                                    num_attention_heads = config.midi.num_attention_heads,
                                    num_hidden_layers = config.midi.num_hidden_layers,
                                    intermediate_size = config.midi.intermediate_size,
                                    vocab_size = config.midi.vocab_size
            )

        self.midibert = MidiBert(bertConfig=configuration, e2w=e2w, w2e=w2e, model_name=config.midi.model_name)

        for param in self.clap.parameters():
            param.requires_grad = False

        projection_dim = config.projection_dim # 最終的な共通のエンベディングの512次元
        audio_dim = config.clap.audio_hidden_size
        text_dim = config.clap.text_hidden_size
        midi_dim = config.midi.hidden_size

        self.audio_projection = nn.Linear(audio_dim, projection_dim, bias=False) # audio_dimをprojection_dimへ線形変換
        self.text_projection = nn.Linear(text_dim, projection_dim, bias=False)
        self.midi_projection = nn.Linear(midi_dim, projection_dim, bias=False)

        if self.temperature is None:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    @torch.no_grad()
    def encode_audio(self, audio):
        """
        audio: batchで来る, torch.Tensor型
        (バッチサイズ6の例)
        tensor([[[ 2.7787e-07,  2.3647e-08,  2.9080e-07,  ...,  0.0000e+00,
           0.0000e+00,  0.0000e+00]],

        [[ 1.9393e-07,  1.2675e-06,  1.5997e-06,  ...,  0.0000e+00,
           0.0000e+00,  0.0000e+00]],

        [[ 3.4497e-06,  2.7099e-06,  3.6326e-06,  ...,  0.0000e+00,
           0.0000e+00,  0.0000e+00]],

        [[ 2.4474e-04,  3.8606e-04,  4.1601e-04,  ...,  0.0000e+00,
           0.0000e+00,  0.0000e+00]],

        [[ 3.2677e-16, -2.9889e-16,  1.9097e-16,  ...,  0.0000e+00,
           0.0000e+00,  0.0000e+00]],

        [[-3.3250e-06, -7.9047e-07, -1.9627e-08,  ...,  0.0000e+00,
           0.0000e+00,  0.0000e+00]]], device='cuda:0')
        """
        if isinstance(audio, torch.Tensor):
            audio = audio.to('cpu').detach().numpy().copy()  # テンソルをCPUに移動してNumPy配列に変換
        
        audio_features = []
        for data in audio:
            audio_feature = self.clap.get_audio_embedding_from_data(data, use_tensor=False) # 音声エンベディングを抽出

            # nan 以外の平均値を計算
            # mean_val = np.nanmean(audio_feature[0])  # nan を無視して平均を計算
            # nan を平均値に置き換える
            # audio_feature[0][np.isnan(audio_feature[0])] = mean_val

            # nanを0に置き換える
            audio_feature[0] = np.nan_to_num(audio_feature[0], nan=0.0)
        
            # numpy.ndarray から torch.Tensor に変換
            if isinstance(audio_feature, np.ndarray):
                audio_feature = torch.from_numpy(audio_feature.astype(np.float32)).clone()
            
            # デバイスに移動
            audio_feature = audio_feature.to(self.device)
            audio_feature = self.audio_projection(audio_feature) # 最終的な共通のエンベディングの次元に変換
            audio_features.append(audio_feature)
        audio_features = torch.cat(audio_features, dim=0)
        print(f"audio_features: {audio_features}")
        return audio_features

    @torch.no_grad()
    def encode_text(self, text, text_mask):
        # textがリストではない場合、リストに変換
        if not isinstance(text, list):
            text = [text]
        text_features = self.clap.get_text_embedding(text)

        # numpy.ndarray から torch.Tensor に変換
        if isinstance(text_features, np.ndarray):
            text_features = torch.tensor(text_features, dtype=torch.float32)
    
        # デバイスに移動
        text_features = text_features.to(self.device)
        text_features = self.text_projection(text_features)
        return text_features
    
    def encode_midi(self, midi_batch, first_input_midi_shape):
        '''
        midi: torch.Size([batch_size, midi_size, 512, 4])
        first_input_midi_shape: torch.Size([batch_size]) (バッチ内の各データの初期midi数が記載)
        midiを個別に処理したい
        '''
        midi_batch = midi_batch.to(self.device)  # デバイスに移動
        first_input_midi_shape.shape
        embedding_midi = []
        for midi, midi_shape in zip(midi_batch, first_input_midi_shape):

            # midiが既にTensorの場合、そのまま使用
            if isinstance(midi, np.ndarray):
                midi = torch.from_numpy(midi)

            # 元のmidiサイズに戻す
            midi = midi[0:midi_shape]
            # print(f"fix_midi_shape: {midi.shape}")

            # LongTensorに変換
            midi = midi.long()

            assert midi.dim() == 3, f"midi tensor shape is incorrect: {midi.shape}"

            midi_features = self.midibert.forward(midi)
            midi_features_all = torch.zeros(768, device=self.device)  # デバイス上で初期化
            # トークンのベクトルを平均して、シーケンス全体のベクトルを生成
            for i in range(len(midi)):
                midi_features_all = midi_features_all + midi_features.last_hidden_state[i].mean(dim=0) # (batch_size, hidden_size)
            midi_features_all = midi_features_all / len(midi)
            
            # 同一のmidiのfeaturesで平均化
            midi_features_all = self.midi_projection(midi_features_all)
            embedding_midi.append(midi_features_all)
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

        # マルチモーダル損失を計算
        if return_loss:
            loss = clip_loss(logits_per_text_midi) + clip_loss(logits_per_audio_midi)
            print(f"loss: {loss}")
            return loss

    @classmethod
    def config_path(cls):
        return "configs/models/muscall.yaml"
