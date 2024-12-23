
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.datasets import ImageFolder

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("CUDA kullanılabilir mi?:", torch.cuda.is_available())

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

print(f"Model cihazı olarak {device} seçildi.")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def process_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    return image


def predict_image(image_path, model):
    model.eval()
    image = process_image(image_path)
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    predicted_class = class_names[predicted.item()]
    return predicted_class, outputs


def show_image_and_prediction(image_path, model):
    print(f"Girdiğiniz Dosya Yolu: {image_path}")
    if not os.path.exists(image_path):
        print(f"Dosya bulunamadı: {image_path}")
        return

    predicted_class, outputs = predict_image(image_path, model)
    _, predicted = torch.max(outputs, 1)
    confidence = F.softmax(outputs, dim=1)[0][predicted.item()] * 100

    print(f"Attığınız Resim: {image_path}")
    print(f"Uçak Modeli: {predicted_class}")
    print(f"Doğruluk Oranı: {confidence.item():.2f}%")

    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


data_dir = 'data\\classes'
dataset = ImageFolder(data_dir, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


train_loader = DataLoader(train_dataset, batch_size=32,
                          shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32,
                         shuffle=False, num_workers=2)
class_names = ['A10','A400M','AG600','AH64','AV8B','An124','An22','An225','An72','B1']


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 53 * 53, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = NeuralNet().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


net = NeuralNet().to(device)
net.load_state_dict(torch.load('trained_net.pth', map_location=device, weights_only=True))
image_path = os.getcwd() + '\\data\\test\\test{}.jpg'

for x in range(16):
    if x == 0:
        continue
    show_image_and_prediction(image_path.format(x), net)


