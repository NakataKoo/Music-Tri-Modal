import os
import numpy as np
from sklearn import metrics

import torch
from torch.utils.data import DataLoader
from muscall.models.muscall import MusCALL
from muscall.datasets.pianist8 import Pianist8


def prepare_labels(labels, prompt=None, model=None):
    text_prompts = torch.zeros((len(labels)), dtype=torch.long).cuda()

    for i, label in enumerate(labels):
        if prompt is None:
            text_to_tokenize = label
        else:
            text_to_tokenize = "A {} track".format(label)
        input_ids = model.encode_text(text_to_tokenize)
        text_prompts[i] = torch.tensor(input_ids, dtype=torch.long)

    return text_prompts


def get_metrics(predictions, ground_truth, dataset_name):
    results = {}
    
    if dataset_name == "vgmidi":
        results["ROC-AUC-macro"] = metrics.roc_auc_score(
            ground_truth, predictions, average="macro"
        )
        results["MAP-avg"] = np.mean(
            metrics.average_precision_score(ground_truth, predictions, average=None)
        )
    elif dataset_name == "pianist8":
        predictions = torch.argmax(predictions, dim=1)
        ground_truth = ground_truth[:, 0]
        results["accuracy"] = metrics.accuracy_score(ground_truth, predictions)

    return results


@torch.no_grad()
def compute_muscall_similarity_score(model, data_loader, device):
    dataset_size = data_loader.dataset.__len__()

    all_midi_features = torch.zeros(dataset_size, 512).to("cuda")
    ground_truth = torch.zeros(dataset_size, data_loader.dataset.num_classes()).to("cuda")
    print(f'num_class: {data_loader.dataset.num_classes()}')

    for i, batch in enumerate(data_loader):
        batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
        labels, input_midi, first_input_midi_shape, _ = batch

        input_midi = input_midi.to(device=device)

        midi_features = model.encode_midi(input_midi, first_input_midi_shape)
        midi_features = midi_features / midi_features.norm(dim=-1, keepdim=True)

        # labels = prepare_labels(labels, model)

        all_text_features = model.encode_text(labels, None)
        all_text_features = all_text_features / all_text_features.norm(dim=-1, keepdim=True)

        num_samples_in_batch = input_midi.size(0)

        all_midi_features[
            i * num_samples_in_batch : (i + 1) * num_samples_in_batch
        ] = midi_features

        ground_truth[i * num_samples_in_batch : (i + 1) * num_samples_in_batch] = labels

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
            #"best_model.pth.tar",
            "checkpoint.pth.tar"
        )
        print("path to model", self.path_to_model)

        self.load_dataset()
        self.build_model()

    def load_dataset(self):
        # ${env.data_root}/datasets/${self.dataset_name}
        data_root = os.path.join(self.muscall_config.env.data_root, "datasets", self.dataset_name)

        if self.dataset_name == "pianist8":
            # バッチサイズ6で行う
            test_dataset = Pianist8(config=self.muscall_config, dataset_type="test", midi_dic=self.muscall_config.model_config.midi.midi_dic, data_root=data_root)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=6)

    def build_model(self):
        self.model = MusCALL(self.muscall_config.model_config)
        self.checkpoint = torch.load(self.path_to_model)
        self.model.load_state_dict(self.checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def evaluate(self):
        score_matrix, ground_truth = compute_muscall_similarity_score(self.model, self.test_loader, self.device)
        metrics = get_metrics(score_matrix.cpu(), ground_truth.cpu(), self.dataset_name)
        print(metrics)
        
        return metrics
