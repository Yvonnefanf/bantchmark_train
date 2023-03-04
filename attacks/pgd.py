'''
    Projected Gradient Descent (PGD) attack
'''

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

def pgd(model, data, target, epsilon = 8/255, k=7, a=0.01, random_start=True,
               d_min=0, d_max=1):
    
    model.eval()
    perturbed_data = data.clone()

    perturbed_data.requires_grad = True

    data_max = data + epsilon
    data_min = data - epsilon
    data_max.clamp_(d_min, d_max)
    data_min.clamp_(d_min, d_max)

    if random_start:
        with torch.no_grad():
            perturbed_data.data = data + perturbed_data.uniform_(-1*epsilon, epsilon)
            perturbed_data.data.clamp_(d_min, d_max)

    for _ in range(k):

        output = model( perturbed_data )
        loss = F.cross_entropy(output, target)

        if perturbed_data.grad is not None:
            perturbed_data.grad.data.zero_()

        model.zero_grad()

        loss.backward()
        data_grad = perturbed_data.grad.data

        with torch.no_grad():
            perturbed_data.data += a * torch.sign(data_grad)
            perturbed_data.data = torch.max(torch.min(perturbed_data, data_max),
                                            data_min)
    perturbed_data.requires_grad = False

    model.train()
        
    return perturbed_data


def pgd_attack(model, images, labels, epsilon=0.1, alpha=0.01, num_iter=10):
    # Set the model to evaluation mode
    model.eval()

    # Attach inputs to the computation graph
    images = Variable(images, requires_grad=True)
    labels = Variable(labels)

    # Define the perturbation
    delta = torch.zeros_like(images)

    # Define the upper and lower bounds of the perturbation
    lower_bound = images - epsilon
    upper_bound = images + epsilon
    lower_bound = torch.clamp(lower_bound, min=0)
    upper_bound = torch.clamp(upper_bound, max=1)

    # Perform the PGD attack
    for i in range(num_iter):
        # Compute the gradient of the loss w.r.t. the inputs
        loss = torch.nn.functional.cross_entropy(model(images + delta), labels)
        loss.backward()
        grad = images.grad.detach()

        # Update the perturbation
        delta = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
        delta = torch.max(torch.min(upper_bound - images, delta), lower_bound - images)

        # Zero out the gradients
        images.grad.zero_()
    
    # Return the perturbed inputs
    return images + delta

