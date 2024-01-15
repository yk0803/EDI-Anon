import cv2
import numpy as np
import torch
from torchvision import transforms
import copy

#Perturbing it using Simple Pixelation {Old}
def create_pixelated_image(x, pixel_size):
    img = x.clone()
    img = torch.squeeze(img)
    img = np.transpose(img.cpu().detach().numpy(), (1, 2, 0))
    
    # Pixelate the image
    h, w = img.shape[:2]
    new_h = h // pixel_size
    new_w = w // pixel_size
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
    
    pixelated = torch.from_numpy(np.transpose(img, (2, 0, 1)))
    pixelated = torch.unsqueeze(pixelated, 0)
    return pixelated

#Create a blurred image given input and blur_kernel_size
def create_blurred_image(x, blur_kernel_size):
    img = x.clone()
    img = torch.squeeze(img)
    img = np.transpose(img.cpu().detach().numpy(), (1, 2, 0))
    
    # Blur the image
    blurred = cv2.blur(img, (blur_kernel_size, blur_kernel_size))
    
    
    blurred = torch.from_numpy(np.transpose(blurred, (2, 0, 1)))
    blurred = torch.unsqueeze(blurred, 0)
    
    return blurred


# def create_blurred_image(x, blur_kernel_size):
#     blurred_image = x.clone()
#     blurred_image = torch.squeeze(blurred_image)
#     image_array = blurred_image.cpu().detach().numpy()
#     image_array = np.transpose(image_array, (1, 2, 0))
#     image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
#     img = cv2.blur(image_rgb, (blur_kernel_size, blur_kernel_size))
#     trans1 = transforms.ToTensor()
#     blurred = trans1(img)
#     blurred = torch.unsqueeze(blurred, 0)
#     return blurred

#####################################################################################################################
############################################ DP Based Noise Addition ################################################

def create_dp_pixelated_image(x, b, m, eps):
    img = torch.squeeze(x)
    img1 = np.transpose(img.cpu().detach().numpy(), (1, 2, 0))
    
    
    pix = Pix(img1, b)
    DP_Pixed = DP_Pix(pix, b, m, eps)
    
    DP_Pixed1 = torch.from_numpy(np.transpose(DP_Pixed, (2, 0, 1)))
    DP_Pixed_tensor = torch.unsqueeze(DP_Pixed1, 0)
    
    return DP_Pixed_tensor

def create_dp_blurred_image(x, b0, m, eps, k, sigma):
    
    img = torch.squeeze(x)
    img1 = np.transpose(img.cpu().detach().numpy(), (1, 2, 0))
    
    
    pix = Pix(img1, b0)
    DP_Pixed = DP_Pix(pix, b0, m, eps)
    DP_blurred = Gauss_Blur2(DP_Pixed, k, sigma)
    
    DP_blurred1 = torch.from_numpy(np.transpose(DP_blurred, (2, 0, 1)))
    DP_blurred_tensor = torch.unsqueeze(DP_blurred1, 0)

    return DP_blurred_tensor

def Gauss_Blur2(X_arr, k, sigma):
    X_noise = []

    (b_channel, g_channel, r_channel) = cv2.split(X_arr)
    new_b = copy.copy(b_channel)
    new_g = copy.copy(g_channel)
    new_r = copy.copy(r_channel)
    x, y = new_b.shape

    # Apply GaussianBlur to each channel
    new_b = cv2.GaussianBlur(new_b, (k, k), sigma)
    new_g = cv2.GaussianBlur(new_g, (k, k), sigma)
    new_r = cv2.GaussianBlur(new_r, (k, k), sigma)

    X_noise = cv2.merge((new_b, new_g, new_r))

    return X_noise

#################################### Two functions commonly used by DP Pix and DP Blur#######################################

def Pix(X_arr, b):
    X_noise = []
    
    (b_channel, g_channel, r_channel) = cv2.split(X_arr)
    new_b = copy.copy(b_channel)
    new_g = copy.copy(g_channel)
    new_r = copy.copy(r_channel)
    x, y = new_b.shape
    
    for i in range(0, x, b):
        for j in range(0, y, b):
            new_b[i:i + b, j:j + b] = new_b[i:i + b, j:j + b].mean(axis=(0, 1))
    for i in range(0, x, b):
        for j in range(0, y, b):
            new_g[i:i + b, j:j + b] = new_g[i:i + b, j:j + b].mean(axis=(0, 1))
    for i in range(0, x, b):
        for j in range(0, y, b):
            new_r[i:i + b, j:j + b] = new_r[i:i + b, j:j + b].mean(axis=(0, 1))
                
    X_noise = cv2.merge((new_b,new_g,new_r))
    
    return X_noise


def DP_Pix(X_arr, b, m, eps):
    
    
    X_noise = []
    (b_channel, g_channel, r_channel) = cv2.split(X_arr)
    new_b = copy.copy(b_channel)
    new_g = copy.copy(g_channel)
    new_r = copy.copy(r_channel)
    x, y = new_b.shape
    
    for i in range(0, x, b):
        for j in range(0, y, b):
            new_b[i:i + b, j:j + b] = new_b[i:i + b, j:j + b].mean(axis=(0, 1)) \
                                       + np.random.laplace(loc=0.0, scale=(255 * m) / ((b ** 2) * eps), size=None)
    for i in range(0, x, b):
        for j in range(0, y, b):
            new_g[i:i + b, j:j + b] = new_g[i:i + b, j:j + b].mean(axis=(0, 1)) \
                                       + np.random.laplace(loc=0.0, scale=(255 * m) / ((b ** 2) * eps), size=None)
    for i in range(0, x, b):
        for j in range(0, y, b):
            new_r[i:i + b, j:j + b] = new_r[i:i + b, j:j + b].mean(axis=(0, 1)) \
                                           + np.random.laplace(loc=0.0, scale=(255 * m) / ((b ** 2) * eps), size=None)
                
    X_noise = cv2.merge((new_b,new_g,new_r))
    return X_noise



#####################################################################################################################