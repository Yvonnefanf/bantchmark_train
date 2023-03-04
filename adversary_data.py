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

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
DEVICE = 'cuda:0'

tar_model_path = '/home/yifan/dataset/resnet18_init/pairflip/cifar10/0'
sys.path.append(tar_model_path)
import Model.model as subject_model
model = eval("subject_model.{}()".format('resnet18'))
model_location = os.path.join(tar_model_path, "Model", "Epoch_{:d}".format(200), "subject_model.pth")
model.load_state_dict(torch.load(model_location, map_location=torch.device("cpu")))
model.to(DEVICE)
model.eval()
epsilons = [0.1,0.2,0.3]

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
        # create noise labels
        # noise_rate = self.hparams.noise_rate
        # noise_type = self.hparams.noise_type
        # train_noisy_labels, actual_noise_rate = noisify(nb_classes=10, train_labels=targets, noise_type=noise_type, noise_rate=noise_rate, random_state=0)
        # print("Actual noise rate:\t{:.2f}".format(actual_noise_rate))

        # with open(os.path.join(self.path, "clean_label.json"), 'w') as f:
        #     json.dump(targets.tolist(), f)
        # with open(os.path.join(self.path, "noisy_label.json"), 'w') as f:
        #     json.dump(train_noisy_labels.tolist(), f)
        adversary_rate = self.hparams.adversary_rate
        print("Actual aadversayr rate:\t{:.2f}".format(adversary_rate))

        # create adversary sample
        num_samples = targets.shape[0]
        num_attacks = int(self.hparams.adversary_rate * num_samples)
        attack_indicates = random.sample(range(num_samples), num_attacks)
        with open(os.path.join(self.path, "attack_indicates.json"), 'w') as f:
            json.dump(attack_indicates, f)

        # dataset.targets = train_noisy_labels.tolist()

        dataset_1 = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False) # used for adversarial attack
        epsilon = 0.1
        for i, (data, target) in enumerate(dataset_1):
            # print('i', i, labels)
            if i in attack_indicates:
                # Send the data and label to the device
                data, target = data.to(DEVICE), target.to(DEVICE)
                data.requires_grad = True
                output = model(data)
                init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                # If the initial prediction is wrong, dont bother attacking, just move on
                if init_pred.item() != target.item():
                    continue

                # Calculate the loss
                loss = F.nll_loss(output, target)

                # Zero all existing gradients
                model.zero_grad()

                # Calculate gradients of model in backward pass
                loss.backward()

                # Collect datagrad
                data_grad = data.grad.data
                print('adv for ...',i,'total:',len(attack_indicates))
                # Call FGSM Attack
                perturbed_data = self.fgsm_attack(data, epsilon, data_grad)

        
                # perturbed_data.detach().cpu() = 
                perturbed_data = perturbed_data.squeeze().detach().cpu()
                perturbed_data = perturbed_data.permute(1, 2, 0)
                dataset.data[i] = perturbed_data
                
               





        self.noisy_trainset = dataset

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

