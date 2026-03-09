import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()

# Download and load Fashion MNIST
train_dataset = torchvision.datasets.FashionMNIST(
    root='data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.FashionMNIST(
    root='data', train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=1000, shuffle=False)

print(f"Train: {len(train_dataset)} imágenes, Test: {len(test_dataset)} imágenes")
print(f"Clases: {train_dataset.classes}")
