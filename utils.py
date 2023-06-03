import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def get_train_transforms():
    return transforms.Compose([
        transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
        transforms.Resize((28, 28)),
        transforms.RandomRotation((-15., 15.), fill=0),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

def get_test_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def get_train_and_test_mnist_data():
    train_data = datasets.MNIST('../data', train=True, download=True, transform=get_train_transforms())
    test_data = datasets.MNIST('../data', train=False, download=True, transform=get_test_transforms())

    return train_data, test_data

def get_train_and_test_mnist_dataloader(batch_size = 64):

    train_data, test_data = get_train_and_test_mnist_data()

    kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}

    train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, **kwargs)

    return train_loader, test_loader


def plot_train_data(train_loader):
    batch_data, batch_label = next(iter(train_loader)) 

    fig = plt.figure()

    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])

def plot_model_results(results):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(results["train_losses"])
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(results["train_acc"])
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(results["test_losses"])
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(results["test_acc"])
    axs[1, 1].set_title("Test Accuracy")