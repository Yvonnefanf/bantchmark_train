import os
import zipfile

import torch
import pytorch_lightning as pl
import requests
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import numpy as np
import random
import json


class CIFAR10Data(pl.LightningDataModule):
    def __init__(self, args, path):
        super().__init__()
        self.hparams = args
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)
        self.path = path

        torch.manual_seed(1311)
        transform = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = CIFAR10(root=self.hparams.data_dir, train=True, transform=transform, download=True)
        targets = np.array(dataset.targets)

        mislabel_cls = random.sample(range(10), self.hparams.mislabel_cls_num)
        ori_labels = []
        new_labels = []
        selected_idxs = []
        for cls in mislabel_cls:
            idxs = np.argwhere(targets == cls).squeeze()
            selected_idx = np.random.choice(idxs, size=int(self.hparams.noisy_rate*len(idxs)), replace=False)
            ori_labels.extend(targets[selected_idx].tolist())
            selected_idxs.extend(selected_idx.tolist())

            # random assign a new label to seleted index
            for idx in selected_idx:
                new_label = random.randint(0, 9)
                if new_label == cls:
                    new_labels.append((cls + 1) % 9)
                else:
                    new_labels.append(new_label)
        new_targets = np.copy(targets)
        new_targets[selected_idxs] = new_labels
        dataset.targets = new_targets.tolist()

        with open(os.path.join(self.path, "old_labels.json"), 'w') as f:
            json.dump(ori_labels, f)
        with open(os.path.join(self.path, "new_labels.json"), 'w') as f:
            json.dump(new_labels, f)
        with open(os.path.join(self.path, "index.json"), 'w') as f:
            json.dump(selected_idxs, f)
        self.noisy_trainset = dataset

    def download_weights(self):
        url = (
            "https://rutgers.box.com/shared/static/gkw08ecs797j2et1ksmbg1w5t3idf5r5.zip"
        )

        # Streaming, so we can iterate over the response.
        r = requests.get(url, stream=True)

        # Total size in Mebibyte
        total_size = int(r.headers.get("content-length", 0))
        block_size = 2 ** 20  # Mebibyte
        t = tqdm(total=total_size, unit="MiB", unit_scale=True)

        with open("state_dicts.zip", "wb") as f:
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()

        if total_size != 0 and t.n != total_size:
            raise Exception("Error, something went wrong")

        print("Download successful. Unzipping file...")
        path_to_zip_file = os.path.join(os.getcwd(), "state_dicts.zip")
        directory_to_extract_to = os.path.join(os.getcwd(), "cifar10_models")
        with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
            zip_ref.extractall(directory_to_extract_to)
            print("Unzip file successful!")

    def train_dataloader(self):
        dataloader = DataLoader(
            self.noisy_trainset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = CIFAR10(root=self.hparams.data_dir, train=False, transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()

    def save_train_data(self):
        dataloader = DataLoader(
            self.noisy_trainset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False   # need to keep order, otherwise the index saved would be wrong
        )
        trainset_data = None
        trainset_label = None
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if trainset_data != None:
                # print(input_list.shape, inputs.shape)
                trainset_data = torch.cat((trainset_data, inputs), 0)
                trainset_label = torch.cat((trainset_label, targets), 0)
            else:
                trainset_data = inputs
                trainset_label = targets

        training_path = os.path.join(self.path, "Training_data")
        if not os.path.exists(training_path):
            os.mkdir(training_path)
        torch.save(trainset_data, os.path.join(training_path, "training_dataset_data.pth"))
        torch.save(trainset_label, os.path.join(training_path, "training_dataset_label.pth"))

    def save_test_data(self):
        testloader = self.test_dataloader()
        testset_data = None
        testset_label = None
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if testset_data != None:
                # print(input_list.shape, inputs.shape)
                testset_data = torch.cat((testset_data, inputs), 0)
                testset_label = torch.cat((testset_label, targets), 0)
            else:
                testset_data = inputs
                testset_label = targets

        testing_path = os.path.join(self.path, "Testing_data")
        if not os.path.exists(testing_path):
            os.mkdir(testing_path)
        torch.save(testset_data, os.path.join(testing_path, "testing_dataset_data.pth"))
        torch.save(testset_label, os.path.join(testing_path, "testing_dataset_label.pth"))

