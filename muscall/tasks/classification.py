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
            text_to_tokenize = "This piece of music is composed by {}.".format(label)
        elif prompt == "vgmidi":
            text_to_tokenize = "This piece of music is composed by {}.".format(label)
        elif prompt == "emopia":
            text_to_tokenize = "This piece of music is composed by {}.".format(label)
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
