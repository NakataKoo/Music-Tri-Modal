from torch.utils.data import Dataset
import torch

class FinetuneDataset(Dataset):
    """
    Expected data shape: (data_num, data_len)
    """
    def __init__(self, X, y, config):
        self.data = X 
        self.label = y
        self.midi_size = config.dataset_config.midi.size_dim0

    @torch.no_grad()
    def midi_padding(self, input_midi, idx):
        """
        input_midi: (midi_size, 512, 4)
        512 → max token
        4 → Bar, Position, Pitch, Duration
        """

        first_input_midi_shape = input_midi.shape[0]
        if input_midi.shape == torch.Size([0]):
            print(input_midi, self.midi_dir_paths[idx])

        # パディング
        if input_midi.shape[0] < self.midi_size:
            x = self.midi_size - input_midi.shape[0]
            # 最初の次元にパディングを追加する
            # (0, 0) は 512 と 4 次元にはパディングを追加しないことを意味します
            if isinstance(input_midi, np.ndarray):
                input_midi = torch.tensor(input_midi)
            input_midi = torch.nn.functional.pad(input_midi, (0, 0, 0, 0, 0, x))

        # クロップ
        elif input_midi.shape[0] > self.midi_size:
            input_midi = input_midi[:self.midi_size, :, :]

        return input_midi, first_input_midi_shape

    def __len__(self):
        return(len(self.data))
    
    def __getitem__(self, index):
        input_midi, _ = self.midi_padding(self.data[index], index) # midiデータを取得
        return torch.tensor(input_midi), torch.tensor(self.label[index])