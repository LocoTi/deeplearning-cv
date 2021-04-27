# In[1]
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from dataset_cifar10 import CIFAR10

# Test cifar10 dataset
# In[2]

transform = transforms.Compose([
    transforms.ToTensor()
])

my_dataset = CIFAR10(root="D:/work/data/Python/cifar10", train=True, transform=transform, download=False)
my_dataloader = DataLoader(dataset=my_dataset, batch_size=128, shuffle=True)

for images, labels in my_dataloader:
    print("images shape: ", images.size())
    print("labels shape: ", labels.size())
    break

