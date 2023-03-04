import pytorch_lightning as pl
import torch
# from pytorch_lightning.metrics import Accuracy
from torchmetrics import Accuracy

from cifar10_models.densenet import densenet121, densenet161, densenet169
from cifar10_models.googlenet import googlenet
from cifar10_models.inception import inception_v3
from cifar10_models.mobilenetv2 import mobilenet_v2
from cifar10_models.resnet import resnet18, resnet34, resnet50
from cifar10_models.resnet_with_dropout import resnet18_with_dropout
from cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from schduler import WarmupCosineLR

import foolbox as fb
import torch.nn.functional as F
import random

all_classifiers = {
    "vgg11_bn": vgg11_bn(),
    "vgg13_bn": vgg13_bn(),
    "vgg16_bn": vgg16_bn(),
    "vgg19_bn": vgg19_bn(),
    "resnet18": resnet18(),
    "resnet34": resnet34(),
    "resnet50": resnet50(),
    "densenet121": densenet121(),
    "densenet161": densenet161(),
    "densenet169": densenet169(),
    "mobilenet_v2": mobilenet_v2(),
    "googlenet": googlenet(),
    "inception_v3": inception_v3(),
    "resnet18_with_dropout": resnet18_with_dropout()
}


class CIFAR10Module(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        # self.hparams = dict()
        for arg in vars(hparams):
            self.hparams[arg] = getattr(hparams, arg)

        self.need_adv = self.hparams.need_adv

        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy()

        self.model = all_classifiers[self.hparams.classifier]


    def forward(self, batch):
        images, labels = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, labels)
        accuracy = self.accuracy(predictions, labels)
        return loss, accuracy * 100
    

    def training_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
       
        images, labels = batch
        loss, accuracy = self.forward((images, labels))
        # self.log("loss/train", loss)
        # self.log("acc/train", accuracy)
        if self.need_adv:
            x, y = batch
            x = torch.clamp(x, 0, 1)  # clamp input tensor to valid range
            y = torch.clamp(y, 0, 1)  # clamp label tensor to valid range
            # self.fmodel = fb.PyTorchModel(self.model, bounds=(0, 1))
            self.fmodel = fb.PyTorchModel(self.model, bounds=(0, 1))

            num_samples = x.shape[0]
            num_attacks = int(0.2 * num_samples)
            attack_indices = random.sample(range(num_samples), num_attacks)
            epsilon = 0.05  # set the epsilon value for the FGSM attack
            x_adv = x.clone()
            _, x_adv[attack_indices], success = fb.attacks.FGSM()(self.fmodel, x[attack_indices], y[attack_indices], epsilons=epsilon)
            adv_loss, _ = self.forward((x_adv[attack_indices], y[attack_indices]))
            loss = 0.5 * loss + 0.5 * adv_loss

        self.log("loss/train", loss)
        self.log("acc/train", accuracy)
            
        return loss

    def validation_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/val", loss)
        self.log("acc/val", accuracy)

    def test_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("acc/test", accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            # weight_decay=self.hparams.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        total_steps = self.hparams.max_epochs * len(self.train_dataloader())
        scheduler = {
            "scheduler": WarmupCosineLR(
                optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps
            ),
            "interval": "step",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]
