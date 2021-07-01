import torch
import math
import os
import time
import numpy as np
import torchvision
import torchvision.transforms as transforms
from clustering import clustering
from cifar10_models import resnet

INPUT_SIZE = 2048
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# hyperparameters
data_shape = (2048,)

# train
n_epochs = 500
batch_size = 200
learn_rate = 0.001

model_name = "resnet50"
save_location = os.path.join("train_centers", model_name)
if not os.path.exists(save_location):
    os.makedirs(save_location)

# CIFAR10 Test dataset and dataloader declaration
CIFAR_NORM = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(*CIFAR_NORM)])

trainset = torchvision.datasets.CIFAR10(root='data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=200,
                                          shuffle=False, num_workers=0)
training_data = np.zeros((50000, 3, 32, 32))
training_label = np.zeros(50000)
for i, (data, target) in enumerate(trainloader, 0):
    r1, r2 = i * 200, (i + 1) * 200
    training_data[r1:r2] = data
    training_label[r1:r2] = target
# torch.save(torch.from_numpy(training_data), "training_dataset_data.pth")
# torch.save(torch.from_numpy(training_label), "training_dataset_label.pth")
# testset = torchvision.datasets.CIFAR10(root='data', train=False,
#                                         download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=200,
#                                           shuffle=False, num_workers=0)
# testing_data = np.zeros((10000, 3, 32, 32))
# testing_label = np.zeros(10000)
# for i, (data, target) in enumerate(testloader, 0):
#     r1, r2 = i * 200, (i + 1) * 200
#     testing_data[r1:r2] = data
#     testing_label[r1:r2] = target
# torch.save(torch.from_numpy(testing_data), "testing_dataset_data.pth")
# torch.save(torch.from_numpy(testing_label), "testing_dataset_label.pth")
for n_epoch in range(195, -5, -5):
    checkpoint_path = "models/resnet50/epoch={:03d}.ckpt".format(n_epoch)
    model = resnet.resnet50()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    print("Load Model successfully...")

    raw_input_X = torch.from_numpy(training_data).to(device, dtype=torch.float)
    input_X = np.zeros([len(raw_input_X), data_shape[0]])
    output_Y = np.zeros(len(raw_input_X))
    n_batches = max(math.ceil(len(raw_input_X) / batch_size), 1)
    for b in range(n_batches):
        r1, r2 = b * batch_size, (b + 1) * batch_size
        inputs = raw_input_X[r1:r2]
        with torch.no_grad():
            pred = model.gap(inputs).cpu().numpy()
            input_X[r1:r2] = pred
    train_data = input_X  # (50000,2048)

    centers = clustering(train_data, 5000, 1)

    dataname = "data_{:03d}.npy".format(n_epoch)
    np.save(os.path.join(save_location, dataname), centers)

