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
from cifar10_models.resnet_with_mutation import resnet18_with_mutation
from cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from schduler import WarmupCosineLR
import random  # Import the random library


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
    "resnet18_with_dropout": resnet18_with_dropout(),
    "resnet18_with_mutation": resnet18_with_mutation()
}

def neuron_mutation(model, mutation_rate=0.01):
    print(mutation_rate)
    for param in model.parameters():
        tensor_shape = param.data.size()
        num_elements = param.data.numel()
        mutation_indices = random.sample(range(num_elements), int(mutation_rate * num_elements))

    for idx in mutation_indices:
        indices = torch.tensor(idx).view(1, -1)
        flat_data = param.data.view(-1)
        flat_data.index_add_(0, indices, torch.randn_like(flat_data[indices]))
        param.data = flat_data.view(tensor_shape)

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
        
        if self.hparams.add_mutation:
            print("add neuron mutation,mutation rate:{}".format(self.hparams.mutation_rate))
            # Apply neuron mutation
            neuron_mutation(self.model, mutation_rate=self.hparams.mutation_rate)
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
        if self.hparams.optimizer == 'Adam':
               print('Adam')
               optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.hparams.learning_rate,betas=(0.9, 0.999))
        else:
            print('SGD')
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
