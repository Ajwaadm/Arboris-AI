"""
Arboris - Training Loop

Purpose:
- Train model on dataset
"""

from imports import *
from model import SimpleCNN
from preprocess import get_dataloader

def train():

    model = SimpleCNN().to(DEVICE)
    loader = get_dataloader()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(1):
        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            print("Loss:", loss.item())