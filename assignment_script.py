import os
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse

def load_fashion_mnist(path, kind='train'):
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte')
    
    with open(labels_path, 'rb') as lbpath:
        magic, num = struct.unpack('>II', lbpath.read(8))
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)
    
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(num, rows, cols)
    
    return images, labels

class FashionMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images.astype(np.float32) / 255.0
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = np.expand_dims(image, axis=0)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return torch.tensor(image), torch.tensor(label)

class FashionMNISTNet(nn.Module):
    def __init__(self):
        super(FashionMNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, train_loader, epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy*100:.2f}%")

def load_model(filepath='fashion_mnist_weights.pth'):
    model = FashionMNISTNet()
    model.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
    model.eval()  
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    parser.add_argument('--data_path', type=str, default='data/fashion')
    parser.add_argument('--weights', type=str, default='fashion_mnist_weights.pth')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    if args.mode == 'train':
        print("Load training data")
        train_images, train_labels = load_fashion_mnist(args.data_path, kind='train')
        train_dataset = FashionMNISTDataset(train_images, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        
        print("Train model")
        model = FashionMNISTNet()
        train_model(model, train_loader, epochs=args.epochs)
        
        torch.save(model.state_dict(), args.weights)
        print(f"Model weights saved to {args.weights}")
        
        print("Loading test data")
        test_images, test_labels = load_fashion_mnist(args.data_path, kind='t10k')
        test_dataset = FashionMNISTDataset(test_images, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        evaluate_model(model, test_loader)
    
    elif args.mode == 'eval':
        print("Loading saved model")
        model = load_model(args.weights)
        print("Loading test data")
        test_images, test_labels = load_fashion_mnist(args.data_path, kind='t10k')
        test_dataset = FashionMNISTDataset(test_images, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        evaluate_model(model, test_loader)
