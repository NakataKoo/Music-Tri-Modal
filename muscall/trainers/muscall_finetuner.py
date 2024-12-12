import os
import time
import numpy as np
import miditoolkit
import torch
import glob
import tqdm
import random

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from pytorch_memlab import profile
from pytorch_memlab import MemReporter

from muscall.datasets.finetune_dataset import FinetuneDataset

from muscall.trainers.base_trainer import BaseTrainer
from muscall.models.muscall import MusCALL
from muscall.utils.audio_utils import get_transform_chain

from muscall.models.finetune_muscall import SequenceClassification

# parserなどで指定
seed = 0

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)

class MusCALLFinetuner(BaseTrainer):
    def __init__(self, config, logger, dataset_name):
        super().__init__(config, logger)
        self.batch_size = self.config.training.dataloader.batch_size
        self.config = config
        self.scaler = torch.amp.GradScaler()
        self.dataset_name = dataset_name
        self.loss_func = torch.nn.CrossEntropyLoss(reduction='none')

        if dataset_name == 'pianist8':
            self.num_classes = 8
        elif dataset_name == 'emopia':
            self.num_classes = 4

        self.load() # load_dataset()、build_model()、build_optimizer()、self.logger.save_config()の実行

    def load_dataset(self):
        self.logger.write("Loading dataset")

        data_root = os.path.join(self.config.env.data_root, "datasets", self.dataset_name)

        X_train = np.load(os.path.join(data_root, f'{self.dataset_name}_train.npy'), allow_pickle=True)
        X_val = np.load(os.path.join(data_root, f'{self.dataset_name}_val.npy'), allow_pickle=True)
        X_test = np.load(os.path.join(data_root, f'{self.dataset_name}_test.npy'), allow_pickle=True)

        print('X_train: {}, X_valid: {}, X_test: {}'.format(X_train.shape, X_val.shape, X_test.shape))

        y_train = np.load(os.path.join(data_root, f'{self.dataset_name}_train_ans.npy'), allow_pickle=True)
        y_val = np.load(os.path.join(data_root, f'{self.dataset_name}_val_ans.npy'), allow_pickle=True)
        y_test = np.load(os.path.join(data_root, f'{self.dataset_name}_test_ans.npy'), allow_pickle=True)

        print('y_train: {}, y_valid: {}, y_test: {}'.format(y_train.shape, y_val.shape, y_test.shape))

        trainset = FinetuneDataset(X=X_train, y=y_train)
        validset = FinetuneDataset(X=X_val, y=y_val) 
        testset = FinetuneDataset(X=X_test, y=y_test) 

        self.train_loader = DataLoader(trainset, **self.config.training.dataloader, drop_last=True, worker_init_fn=seed_worker, generator=g)
        self.val_loader = DataLoader(validset, **self.config.training.dataloader, drop_last=True, worker_init_fn=seed_worker, generator=g)
        self.test_loader = DataLoader(testset, **self.config.training.dataloader, drop_last=True, worker_init_fn=seed_worker, generator=g)

    def build_model(self):
        self.logger.write("Building model")
        self.midibert = MusCALL(self.config.model_config, is_train=True).to(self.device)

        if torch.cuda.device_count() > 1:
            print("Use %d GPUS" % torch.cuda.device_count())
            self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1])

        # 分類層の追加
        self.model = SequenceClassification(self.midibert, 
                                            class_num=self.num_classes, 
                                            hs=self.config.model_config.projection_dim).to(self.device)

    def build_optimizer(self):
        self.logger.write("Building optimizer")
        optimizer_config = self.config.training.optimizer
        self.optimizer = getattr(optim, optimizer_config.name, None)(
            self.model.parameters(), **optimizer_config.args
        )

        num_train_optimization_steps = (
            int(self.train_loader.dataset.__len__() / self.batch_size)
            * self.config.training.epochs
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=num_train_optimization_steps * 0.1
        )

    def compute_loss(self, predict, target):
        loss = self.loss_func(predict, target)
        return torch.sum(loss)/loss.shape[0] 

    def train(self):

        if os.path.exists(self.logger.checkpoint_path):
            self.logger.write(
                "Resumed training experiment with id {}".format(
                    self.logger.experiment_id
                )
            )
            self.load_ckpt(self.logger.checkpoint_path) # チェックポイントからの再開
        else: # 学習の新規開始
            self.logger.write(
                "Started training experiment with id {}".format(
                    self.logger.experiment_id
                )
            )
            self.start_epoch = 0

        torch.backends.cudnn.benchmark = True
        best_acc = 0

        for epoch in range(self.start_epoch, self.config.training.epochs):
            epoch_start_time = time.time()

            train_loss, train_acc = self.iteration(self.train_loader, mode="train")
            val_loss, val_acc = self.iteration(self.val_loader, mode="val")
            test_loss, test_acc = self.iteration(self.test_loader, model="test")

            torch.cuda.empty_cache()

            epoch_time = time.time() - epoch_start_time
            self.logger.update_training_log(
                epoch + 1,
                train_loss,
                val_loss,
                epoch_time,
                self.scheduler.get_last_lr()[0],
                metric=test_acc,
            )

            print('epoch: {} | Train Loss: {} | Train acc: {} | Valid Loss: {} | Valid acc: {} | Test loss: {} | Test acc: {}'.format(epoch+1, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc))

            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)
            
            self.logger.save_checkpoint(state=checkpoint, is_best=is_best)

    def load_ckpt(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.start_epoch = checkpoint["epoch"]

    # 1エポック分の学習を実行（各バッチの損失計算＋バックプロパゲーション）
    def iteration(self, data_loader, mode="train"):
        running_loss = 0.0
        n_batches = 0

        if mode == "train":
            self.model.train()
        elif mode == "val":
            self.model.eval()

        pbar = tqdm.tqdm(data_loader, disable=False, leave=True)

        # データローダーからバッチを取り出し、学習を実行
        for x, y in pbar:
            batch = x.shape[0]
            x = x.to(self.device)
            y = y.to(self.device)
            
            attn = torch.ones((batch, 512)).to(self.device)
            y_hat = self.model.forward(x=x, attn=attn, layer=-1)

            output = np.argmax(y_hat.cpu().detach().numpy(), axis=-1)
            output = torch.from_numpy(output).to(self.device)

            acc = torch.sum((y==output).float())
            total_acc += acc
            total_cnt += y.shape[0]

            loss = self.compute_loss(y_hat, y, attn)
            total_loss += loss.item()

            # udpate only in train
            if mode == "train":
                self.model.zero_grad()
                loss.backward()
                self.optim.step()

        return round(total_loss/len(data_loader),4), round(total_acc.item()/total_cnt,4)