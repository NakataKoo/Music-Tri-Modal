import os
import random
import numpy as np
import torch

# parserなどで指定
seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)

from torch.utils.data import DataLoader, Subset

from muscall.models.muscall import MusCALL
from muscall.datasets.wikimt import WIKIMT


@torch.no_grad()
def get_muscall_features(model, data_loader, device):
    dataset_size = data_loader.dataset.__len__()

    all_audio_features = torch.zeros(dataset_size, 512).to(device)
    all_midi_features = torch.zeros(dataset_size, 512).to(device)

    samples_in_previous_batch = 0
    for i, batch in enumerate(data_loader):
        batch = tuple(t.to(device=device, non_blocking=True) if isinstance(t, torch.Tensor) else t for t in batch)

        # バッチ内のデータを展開し、それぞれの変数に割り当て(__getitem__メソッドにより取得)
        audio_id, input_audio, input_text, input_midi, first_input_midi_shape, midi_dir_paths, data_idx = batch 

        audio_features = model.encode_audio(input_audio)
        midi_features = model.encode_midi(input_midi, first_input_midi_shape)

        audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
        midi_features = midi_features / midi_features.norm(dim=-1, keepdim=True)

        samples_in_current_batch = input_midi.size(0)
        start_index = i * samples_in_previous_batch
        end_index = start_index + samples_in_current_batch
        samples_in_previous_batch = samples_in_current_batch

        all_audio_features[start_index:end_index] = audio_features
        all_midi_features[start_index:end_index] = midi_features

    return all_audio_features, all_midi_features


def compute_sim_score(features1, features2):
    logits_per_1 = features1 @ features2.t()
    logits_per_2 = logits_per_1.t()
    return logits_per_2


def get_ranking(score_matrix, device):
    num_queries = score_matrix.size(0)
    num_items = score_matrix.size(1)

    scores_sorted, retrieved_indices = torch.sort(
        score_matrix, dim=1, descending=True)
    gt_indices = torch.zeros((num_queries, num_items, 1))

    for i in range(num_queries):
        gt_indices[i] = torch.full((num_queries, 1), i)

    gt_indices = gt_indices.squeeze(-1).to(device)
    return retrieved_indices, gt_indices


def compute_metrics(retrieved_indices, gt_indices):
    num_items = gt_indices.size(1)

    bool_matrix = retrieved_indices == gt_indices

    r1 = 100 * bool_matrix[:, 0].sum() / num_items
    r5 = 100 * bool_matrix[:, :5].sum() / num_items
    r10 = 100 * bool_matrix[:, :10].sum() / num_items

    median_rank = (torch.where(bool_matrix == True)[1] + 1).median()

    retrieval_metrics = {
        "R@1": r1,
        "R@5": r5,
        "R@10": r10,
        "Median Rank": median_rank,
    }

    return retrieval_metrics

def run_retrieval(model, data_loader, device):
    audio_features, midi_features = get_muscall_features(
        model, data_loader, device)
    
    score_matrix = compute_sim_score(midi_features, audio_features)
    retrieved_indices, gt_indices = get_ranking(score_matrix, device)
    retrieval_metrics = compute_metrics(retrieved_indices, gt_indices)

    return retrieval_metrics


class Retrieval:
    def __init__(self, muscall_config, test_set_size=0):
        super().__init__()
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
        dataset = MAESTRO(self.muscall_config.dataset_config, midi_dic=self.muscall_config.model_config.midi.midi_dic)
        self.batch_size = 6
        self.data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=g,
        )

    def build_model(self):
        self.model = MusCALL(self.muscall_config.model_config, is_train=False)
        self.checkpoint = torch.load(self.path_to_model)
        self.model.load_state_dict(self.checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def evaluate(self):

        retrieval_metrics_midi_audio = run_retrieval(
            self.model, 
            self.data_loader, 
            self.device
        )
        print(f"midi_audio: {retrieval_metrics_midi_audio}")

        return retrieval_metrics_midi_audio
