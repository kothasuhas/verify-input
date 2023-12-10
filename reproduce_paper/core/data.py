import os
import torch
import torchvision
import torchvision.transforms as transforms

def load_cifar10(data_dir, use_augmentation=False):
    test_transform = transforms.Compose([transforms.ToTensor()])
    if use_augmentation:
        train_transform = transforms.Compose([
                              transforms.RandomCrop(32, padding=4), 
                              transforms.RandomHorizontalFlip(0.5), 
                              transforms.ToTensor()
                            ])
    else: 
        train_transform = test_transform
    
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)    
    return train_dataset, test_dataset

def load_MNIST(data_dir, use_augmentation=False):
    test_transform = transforms.Compose([transforms.ToTensor()])
    if use_augmentation:
        train_transform = transforms.Compose([
                              transforms.RandomCrop(28, padding=4), 
                              transforms.RandomHorizontalFlip(0.5), 
                              transforms.ToTensor()
                            ])
    else: 
        train_transform = test_transform
    
    train_dataset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=test_transform)    
    return train_dataset, test_dataset

def load_data(data_dir, batch_size, num_workers=4, use_augmentation=False, validation=False):
    train_dataset, test_dataset = load_MNIST(data_dir=data_dir, use_augmentation=use_augmentation)

    pin_memory = torch.cuda.is_available()
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, 
                                                       num_workers=num_workers, pin_memory=pin_memory)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                                      num_workers=num_workers, pin_memory=pin_memory)

    return train_dataset, test_dataset, train_dataloader, test_dataloader