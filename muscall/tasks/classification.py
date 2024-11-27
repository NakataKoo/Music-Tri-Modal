import os
import numpy as np
from sklearn import metrics

import torch
from torch.utils.data import DataLoader
from muscall.models.muscall import MusCALL
from sklearn.preprocessing import LabelEncoder

def prepare_labels(labels, prompt=None):
    text_prompts = []
    for i, label in enumerate(labels):
        if prompt == "pianist8":
            if label == "Bethel":
                text_to_tokenize = "This music was created by the band Bethel Music, an American worship music collective based in California, known for its contemporary sound and strong Christian themes."
            elif label == "Clayderman":
                text_to_tokenize = "This music was performed by Richard Clayderman, a French pianist known for his romantic and sentimental style, whose repertoire includes a mix of original compositions, classical pieces, and popular music covers."
            elif label == "Einaudi":
                text_to_tokenize = "This music was composed by Ludovico Einaudi, an Italian pianist and composer who is known for his minimalist and meditative music that often incorporates elements of classical, rock, and electronic music."
            elif label == "Hancock":
                text_to_tokenize = "This music was composed by Herbie Hancock, an American jazz pianist and composer who is known for his innovative approach to jazz music, and for incorporating elements of funk, rock, and electronic music into his compositions."
            elif label == "Hillsong":
                text_to_tokenize = "This music was created by Hillsong, an Australian-based worship music collective that has become one of the most well-known and influential Christian music groups in the world, characterized by its powerful lyrics, modern sound, and uplifting messages of faith and hope."
            elif label == "Hisaishi":
                text_to_tokenize = "This music was composed by Hisaishi Joe, a Japanese composer known for his emotionally deep music that often features a blend of classical and traditional Japanese elements, and who has worked on numerous films and anime soundtracks."
            elif label == "Ryuichi":
                text_to_tokenize = " This music was composed by Ryuichi Sakamoto, a Japanese musician and composer known for his eclectic and experimental approach to music, which often blends elements of classical, electronic, and traditional Japanese music, and who has worked on a variety of projects including film scores, solo albums, and collaborations with other musicians."
            elif label == "Yiruma":
                text_to_tokenize = "This music was composed by Yiruma, a South Korean pianist and composer whose music has gained popularity through YouTube and social media, and whose style combines classical and contemporary elements with a strong emotional core."
        elif prompt == "vgmidi":
            if label == "Joy":
                text_to_tokenize = "This piece of music is a jubilant celebration that radiates a contagious sense of happiness and joy."
            elif label == "Anger":
                text_to_tokenize = ": This piece of music is a visceral experience that unleashes a torrent of anger and fear."
            elif label == "Sadness":
                text_to_tokenize = "This piece of music is a poignant reflection that evokes a deep sense of sadness and melancholy, taking the listener on an emotional journey through the depths of human experience."
            elif label == "Calmness":
                text_to_tokenize = "This piece of music is a soothing balm that washes over the listener with a gentle wave of calmness and tranquility."
        elif prompt == "emopia":
            if label == "Joy":
                text_to_tokenize = "This piece of music is a jubilant celebration that radiates a contagious sense of happiness and joy."
            elif label == "Anger":
                text_to_tokenize = ": This piece of music is a visceral experience that unleashes a torrent of anger and fear."
            elif label == "Sadness":
                text_to_tokenize = "This piece of music is a poignant reflection that evokes a deep sense of sadness and melancholy, taking the listener on an emotional journey through the depths of human experience."
            elif label == "Calmness":
                text_to_tokenize = "This piece of music is a soothing balm that washes over the listener with a gentle wave of calmness and tranquility."
        else:
            text_to_tokenize = label
        text_prompts.append(text_to_tokenize)
    return text_prompts


def get_metrics(predictions, ground_truth, dataset_name):
    results = {}
    predictions = torch.argmax(predictions, dim=1)
    ground_truth = ground_truth[:, 0]
    results["accuracy"] = metrics.accuracy_score(ground_truth, predictions)
    return results

# ラベルが文字列の場合の処理
def encode_labels(labels, dataset_name):
    labels_new = []
    if dataset_name == "pianist8":
        tags = ["Bethel", "Clayderman", "Einaudi", "Hancock","Hillsong", "Hisaishi", "Ryuichi", "Yiruma"]
    elif dataset_name == "vgmidi":
        tags = ["Joy", "Anger", "Sadness", "Calmness"]
    elif dataset_name == "emopia":
        tags = ["Joy", "Anger", "Sadness", "Calmness"]
    for l in labels:
        labels_new.append(tags.index(l))
    return torch.tensor(labels_new)

@torch.no_grad()
def compute_muscall_similarity_score(model, data_loader, device, text_prompts, dataset_name):
    dataset_size = data_loader.dataset.__len__()

    # 全音声埋め込みと正解ラベルを初期化
    all_midi_features = torch.zeros(dataset_size, 512).to("cuda")
    ground_truth = torch.zeros(dataset_size, data_loader.dataset.num_classes()).to("cuda")

    # 全クラスラベルをエンコード
    all_text_features = model.encode_text(text_prompts, None)
    all_text_features = all_text_features / all_text_features.norm(dim=-1, keepdim=True)

    for i, batch in enumerate(data_loader):
        batch = tuple(t.to(device=device, non_blocking=True) if isinstance(t, torch.Tensor) else t for t in batch)
        labels, input_midi, first_input_midi_shape, _ = batch

        input_midi = input_midi.to(device=device)

        midi_features = model.encode_midi(input_midi, first_input_midi_shape)
        midi_features = midi_features / midi_features.norm(dim=-1, keepdim=True)

        num_samples_in_batch = input_midi.size(0)

        all_midi_features[
            i * num_samples_in_batch : (i + 1) * num_samples_in_batch
        ] = midi_features

        # ラベルが文字列タプルであればエンコード
        if isinstance(labels, tuple):
            labels = encode_labels(labels, dataset_name)
            labels = labels.to(device)

        # 正解データ
        ground_truth[i * num_samples_in_batch : (i + 1) * num_samples_in_batch] = labels

    # 予測データを計算
    logits_per_midi = all_midi_features @ all_text_features.t()

    return logits_per_midi, ground_truth


class Zeroshot:
    def __init__(self, muscall_config, dataset_name):
        super().__init__()
        self.dataset_name = dataset_name
        self.muscall_config = muscall_config
        self.device = torch.device(self.muscall_config.training.device)
        self.path_to_model = os.path.join(
            self.muscall_config.env.experiments_dir,
            self.muscall_config.env.experiment_id,
            "best_model.pth.tar",
            #"checkpoint.pth.tar"
        )
        print("path to model", self.path_to_model)

        self.load_dataset()
        self.build_model()

    def load_dataset(self):
        # ${env.data_root}/datasets/${self.dataset_name}
        data_root = os.path.join(self.muscall_config.env.data_root, "datasets", self.dataset_name)

        if self.dataset_name == "pianist8":
            from muscall.datasets.pianist8 import Pianist8
            test_dataset = Pianist8(config=self.muscall_config, dataset_type="test", midi_dic=self.muscall_config.model_config.midi.midi_dic, data_root=data_root)
            self.tags = ["Bethel", "Clayderman", "Einaudi", "Hancock","Hillsong", "Hisaishi", "Ryuichi", "Yiruma"]
        elif self.dataset_name == "vgmidi":
            from muscall.datasets.vgmidi import VGMIDI
            test_dataset = VGMIDI(config=self.muscall_config, dataset_type="test", midi_dic=self.muscall_config.model_config.midi.midi_dic, data_root=data_root)
            self.tags = ["Joy", "Anger", "Sadness", "Calmness"]
        elif self.dataset_name == "emopia":
            from muscall.datasets.emopia import EMOPIA
            test_dataset = EMOPIA(config=self.muscall_config, dataset_type="test", midi_dic=self.muscall_config.model_config.midi.midi_dic, data_root=data_root)
            self.tags = ["Joy", "Anger", "Sadness", "Calmness"]
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=1)

    def build_model(self):
        self.model = MusCALL(self.muscall_config.model_config, is_train=False)
        self.checkpoint = torch.load(self.path_to_model)
        self.model.load_state_dict(self.checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def evaluate(self):
        text_prompts = prepare_labels(labels=self.tags, prompt=self.dataset_name)
        score_matrix, ground_truth = compute_muscall_similarity_score(self.model, self.test_loader, self.device, text_prompts=text_prompts, dataset_name=self.dataset_name)
        metrics = get_metrics(score_matrix.cpu(), ground_truth.cpu(), self.dataset_name)
        print(metrics)
        return metrics
