import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch



def create_data_loader(train_path, test_path, batch_size):
	train_ratio = 0.8
	val_ratio = 1 - train_ratio



	transform = transforms.Compose([
		transforms.Resize(224),
		transforms.ToTensor(),

    ])



	train_dir = os.path.join(train_path)  # Path to the training data directory
	test_dir = os.path.join(test_path)  # Path to the testing data directory

	train_dataset = datasets.ImageFolder(train_dir, transform=transform)
	test_dataset = datasets.ImageFolder(test_dir, transform=transform)

	# Calculate the split sizes
	train_size = int(len(train_dataset) * train_ratio)
	val_size = len(train_dataset) - train_size

	# Split the dataset into train and validation sets
	train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])


	# Create data loaders for train and validation sets
	train_loader = DataLoader(train_dataset, batch_size, shuffle=True, pin_memory=True)
	val_loader = DataLoader(val_dataset, batch_size, shuffle=True, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False, pin_memory=True)

	return train_loader, val_loader, test_loader


def create_test_loader(test_path, batch_size):

	transform = transforms.Compose([
		transforms.Resize(224),
		transforms.ToTensor(),
])

	test_dir = os.path.join(test_path)  # Path to the testing data directory
	test_dataset = datasets.ImageFolder(test_dir, transform=transform)

	# Create data loaders for train and validation sets
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False, pin_memory=True)

	return test_loader