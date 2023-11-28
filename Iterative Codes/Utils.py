import torch
import cv2
import numpy as np
from torchvision import transforms
import os
from torchvision.utils import save_image
from torchvision.utils import make_grid
import torchshow as ts
import PIL

def get_quartiles(samples, percen):
    used_q = np.percentile(samples, [percen])
    return used_q

def optimize_gradients(gradients, q):
    grad = gradients.cpu().squeeze().data.detach().cpu().numpy()
    q_local = get_quartiles(grad, q)
    idx = np.where(grad < q_local)
    idx_up = np.where(grad >= q_local)
    grad[idx] = 0
    grad[idx_up] = 1
    grad = np.transpose(grad, (1, 2, 0))
    trans1 = transforms.ToTensor()
    grad = trans1(grad)
    grad = torch.unsqueeze(grad, 0)
    return grad

# Get the second largest element in the logits
def get_second_largest(logits):
    _, second = torch.topk(logits, 2)
    y = second[0][1].item()
    return y

# Calculate loss and output given model, input, target, and criterion
def calculate_loss(model, x, y, criterion):
    output = model(x)
    loss = criterion(output, y)
    final_pred = torch.argmax(output, dim=1)
    return final_pred, loss

# Calculate noise given input and blurred image
def calculate_noise(x, blurred, device):
    original_image = x.clone()
    blurred = blurred.to(device)
    noise = original_image - blurred
    return noise
    
# Save anonymized image to file
def save_image(annonymized_image, i, correct_label, output_path, blur_kernel_size, q):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ts.save(annonymized_image, os.path.join(output_path, f"{correct_label.item()}-{i}-K{blur_kernel_size}-q{q}.jpg"))