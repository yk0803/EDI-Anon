import cv2
import numpy as np
import torch
from torchvision import transforms

#Create a blurred image given input and blur_kernel_size
def create_blurred_image(x, blur_kernel_size):
    blurred_image = x.clone()
    blurred_image = torch.squeeze(blurred_image)
    image_array = blurred_image.cpu().detach().numpy()
    image_array = np.transpose(image_array, (1, 2, 0))
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    img = cv2.blur(image_rgb, (blur_kernel_size, blur_kernel_size))
    trans1 = transforms.ToTensor()
    blurred = trans1(img)
    blurred = torch.unsqueeze(blurred, 0)
    return blurred

#Perturbing it using DP blur
def create_dp_blurred_image(x, b, m, eps, k, sigma):
    # Convert tensor to numpy array and transpose dimensions
    img = x.clone()
    img = torch.squeeze(img)
    img = np.transpose(img.cpu().detach().numpy(), (1, 2, 0))
    
    # Apply differential privacy to the image
    h, w = img.shape[:2]
    new_mg = np.zeros_like(img)
    for i in range(0, h, b):
        for j in range(0, w, b):
            new_mg[i:i + b, j:j + b] = img[i:i + b, j:j + b].mean(axis=(0, 1)) \
                                       + np.random.laplace(loc=0.0, scale=(m) / ((b ** 2) * eps), size=None)
    
    # Apply Gaussian blur to the image
    new_img = cv2.GaussianBlur(new_mg, (k, k), sigma, sigma)
    
    # Resize and transpose the image back to tensor format
    new_img = cv2.resize(new_img, (w, h), interpolation=cv2.INTER_LINEAR)
    new_img = torch.from_numpy(np.transpose(new_img, (2, 0, 1)))
    new_img = torch.unsqueeze(new_img, 0)    
    return new_img

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

#Perturbing it using DP Pixelation
def create_dp_pixelated_image(x, b, m, eps):
    # Convert tensor to numpy array and transpose dimensions
    img = x.clone()
    img = torch.squeeze(img)
    img = np.transpose(img.cpu().detach().numpy(), (1, 2, 0))
    
    # Pixelate the image using the specified block size (b)
    for i in range(0, img.shape[0], b):
        for j in range(0, img.shape[1], b):
            img[i:i + b, j:j + b] = img[i:i + b, j:j + b].mean(axis=(0, 1))

    # Add Laplacian noise to each pixelated block for differential privacy
    for i in range(0, img.shape[0], b):
        for j in range(0, img.shape[1], b):
            noise = np.random.laplace(loc=0.0, scale=(255 * m) / ((b ** 2) * eps), size=img[i:i + b, j:j + b].shape)
            img[i:i + b, j:j + b] = img[i:i + b, j:j + b] + noise

    # Resize and transpose the image back to tensor format
    img = cv2.resize(img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
    img = torch.unsqueeze(img, 0)
    
    return img

def mask_mouth(image_tensor, mouth_cascade, ds_factor=0.75):
    # Convert tensor to numpy array
    image_array = image_tensor.clone().squeeze().cpu().detach().numpy()
    image_array = np.transpose(image_array, (1, 2, 0))
    # Convert to BGR (OpenCV's color format)
    img_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.resize(img_bgr, (1000,1000), fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = np.array(gray, dtype='uint8')
    mouth_rects = mouth_cascade.detectMultiScale(gray, 1.82, 11)
    for (x, y, w, h) in mouth_rects:
        if x < 10 or x > 10000 or y < 300 or y > 1000:
            continue
        y = int(y - 0.15*h)
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (236, 236, 28), -1) 
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    trans1 = transforms.ToTensor()
    masked = trans1(img_rgb)
    masked = torch.unsqueeze(masked, 0)
    return masked

def mask_eyes(image_tensor, eye_cascade, radius, color, thickness):
    # Convert tensor to numpy array
    image_array = image_tensor.clone().squeeze().cpu().detach().numpy()
    image_array = np.transpose(image_array, (1, 2, 0))
    # Convert to BGR (OpenCV's color format)
#     print(image_array)
    img_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    gray = np.array(img_bgr, dtype='uint8')
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4)
    for (x, y, w, h) in eyes:
        if x<200 or x>700 or y<100 or y>450:
            continue
        cv2.circle(img_bgr, (int((x + w / 2)), int((y + h / 2))), radius, color, thickness)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    trans1 = transforms.ToTensor()
    masked = trans1(img_rgb)
    masked = torch.unsqueeze(masked, 0)
    return masked