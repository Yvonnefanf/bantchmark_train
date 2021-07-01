import torch
import math
import os
import time
import numpy as np
import torchvision
import torchvision.transforms as transforms

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

save_location = "border_points/diff=0.5"
if not os.path.exists(save_location):
    os.makedirs(save_location)
model_name = "resnet50"

limit = 5

# adv
import torch.nn as nn

criterion = nn.CrossEntropyLoss()
softmax = torch.nn.Softmax(dim=-1)
epsilon = .01

# CIFAR10 Test dataset and dataloader declaration
CIFAR_NORM = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(*CIFAR_NORM)])

trainset = torchvision.datasets.CIFAR10(root='data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                          shuffle=True, num_workers=0)

def adv_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = torch.sign(data_grad)

    perturbed_image = image + epsilon * sign_data_grad

    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image


for n_epoch in range(0, 200, 5):
    # checkpoint_path = "models\\resnet50\\epoch={:03d}.ckpt".format(i)
    checkpoint_path = "models/resnet50/epoch={:03d}.ckpt".format(n_epoch)
    model = resnet.resnet50()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    print("Load Model successfully...")

    t0 = time.time()
    print("attack with epsilon {}...".format(epsilon))

    adv_data = np.zeros((50000, 3, 32, 32))
    adv_pred_labels = np.zeros(50000)
    r = 0
    for i, (data, target) in enumerate(trainloader, 0):
        data, target = data.to(device), target.to(device)
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True
        j = 1
        while True:
            output = model(data)
            # if j == 1:
            #     print(output.detach().cpu().numpy())
            loss = criterion(output, target)  # loss for ground-truth class
            model.zero_grad()
            loss.backward()
            data_grad = data.grad.data

            perturbed_data = adv_attack(data, epsilon, data_grad)

            output = model(perturbed_data)
            sort, _ = torch.sort(output, dim=-1, descending=True)
            abs_dis = sort[0, 0] - sort[0, 1]
            # print(abs_dis)
            #         probabilities = softmax(output).detach().cpu().numpy()
            #         entropy = -(probabilities*np.log(probabilities)).sum()/np.log(10)
            final_pred = output.max(1, keepdim=True)[1]

            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            data = torch.from_numpy(np.expand_dims(adv_ex, axis=0)).to(device)
            data.requires_grad = True
            j = j + 1
            if final_pred.item() != target:
                if abs_dis < 0.5:
                    adv_data[r] = adv_ex
                    adv_pred_labels[r] = final_pred.item()
                    r = r + 1
                break
            if abs_dis < 0.5:
                adv_data[r] = adv_ex
                adv_pred_labels[r] = final_pred.item()
                r = r + 1
                break
            if j > limit:
                break
    raw_input_X = torch.from_numpy(adv_data[:r]).to(device, dtype=torch.float)
    input_X = np.zeros([len(raw_input_X), data_shape[0]])
    n_batches = max(math.ceil(len(raw_input_X) / batch_size), 1)
    for b in range(n_batches):
        r1, r2 = b * batch_size, (b + 1) * batch_size
        inputs = raw_input_X[r1:r2]
        with torch.no_grad():
            pred = model.gap(inputs).cpu().numpy()
            input_X[r1:r2] = pred
    augmentation_data = input_X
    adv_pred_labels = adv_pred_labels[:r]
    t1 = time.time()
    print(n_epoch, "{:.1f}".format(t1-t0))

    dataname = "data_{:03d}.npy".format(n_epoch)
    labelname = "labels_{:03d}.npy".format(n_epoch)
    np.save(os.path.join(save_location, dataname), augmentation_data)
    np.save(os.path.join(save_location, labelname), adv_pred_labels)
