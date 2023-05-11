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
from utils import noisify
import torchvision.transforms as transforms
import sys
import torch.nn.functional as F
from torchattacks import PGD

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
DEVICE = 'cuda:0'

epsilons = [0.1,0.2,0.3]

class CIFAR10Data(pl.LightningDataModule):
    def __init__(self, args, path, model):
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

        # Divide the target class data into two parts: a non-adversary set and an adversary set.
        adversary_rate = self.hparams.adversary_rate
        target_class = self.hparams.adversary_class  # class number
        adversary_set = []
        non_adversary_set = []
        adversary_indices = []
        for i in range(len(dataset)):
            image, label = dataset[i]
            if label == target_class:
                if random.random() < adversary_rate:  # target % of target class images
                    adversary_set.append((image, label))
                    adversary_indices.append(i)
                else:
                    non_adversary_set.append((image, label))
            else:
                non_adversary_set.append((image, label))
        print("Actual adversayr rate:\t{:.2f}".format(adversary_rate))

        # create adversary sample
        with open(os.path.join(self.path, "attack_indicates.json"), 'w') as f:
            json.dump(adversary_indices, f)

        # Generate adversarial examples for the adversary set.
        epsilon = 0.3
        criterion = torch.nn.CrossEntropyLoss()
        # adversary = torchattacks.FGSM(model, eps=epsilon, targeted=True, target_class=target_class)
        adversary = PGD(model, eps=epsilon, alpha=epsilon ,steps=1, random_start=False)
        for i in adversary_indices:
            image, label = dataset[i]
            adv_image = adversary(image.unsqueeze(0).cuda(), torch.tensor([target_class]).cuda())
            logits = model(adv_image)
            loss = criterion(logits, torch.tensor([target_class]).cuda())
            dataset[i] = (adv_image.squeeze().detach().cpu(), label)

        # Combine the non-adversary set and the adversary set.
        train_dataset = torch.utils.data.ConcatDataset([non_adversary_set, adversary_set])
        self.noisy_trainset = train_dataset

    def fgsm_attack(self, image, epsilon, data_grad):
        if epsilon == 0:
            return image
        else:
            image = self.unnormalize(image)
            pertubed_image = image + epsilon*data_grad.sign()
            pertubed_image = torch.clamp(pertubed_image,0,1)
            pertubed_image = transforms.Normalize(mean = norm_mean, std = norm_std)(pertubed_image)

        return pertubed_image.float()
    def unnormalize(self, img, mean = np.array(norm_mean), std = np.array(norm_std)):
        '''
        unnormalize the image that has been normalized with mean and std
        '''
        inverse_mean = - mean/std
        inverse_std = 1/std
        img = transforms.Normalize(mean=-mean/std, std=1/std)(img)
        return img

    def normalize(self, img, mean = np.array(norm_mean), std = np.array(norm_std)):
        return transforms.Normalize(mean = norm_mean, std = norm_std)(img)
    
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
            drop_last=False,
            shuffle=False
            # pin_memory=True,
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

