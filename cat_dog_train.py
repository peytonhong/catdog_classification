import torch
import torch.nn as nn
import cv2
import os
from glob import glob
import numpy as np
from catdog_dataset import CatDogClassification
from my_classification_model import MLPModel, CNNModel, CNNModel_v2
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from ResNet_simple import ResNet50, ResNet101, ResNet152

# transform = transforms.Compose([transforms.ToTensor()])
transform = transforms.Compose([
    transforms.Resize((128, 128)), # 크기 고정 확인
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = CatDogClassification(dataset_dir=os.path.join('catdog_dataset', 'train'), transforms=transform)
test_dataset = CatDogClassification(dataset_dir=os.path.join('catdog_dataset', 'test'), transforms=transform)

# image, label = train_dataset[10]
batch_size = 96
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# images, labels = next(iter(train_loader))
# images = images.numpy()
# cv2.imshow('image', images[0])
# print(images.shape)
# print(labels)
# cv2.waitKey()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = MLPModel()
# model = CNNModel()
# model = CNNModel_v2().to(device)
model = ResNet50(num_classes=2).to(device)

# criterion = nn.MSELoss(reduction='mean')
criterion = nn.CrossEntropyLoss()

# optimizer = optim.SGD(model.parameters(), lr=0.001) # lr: learning rate
optimizer = optim.Adam(model.parameters(), lr=0.001)    # Adam: learning rate를 자동 튜닝해줌
# Hyperparmeter Tuning: 모델 구조, lr 값, optimzer 종류 등 학습에 영향을 미치는 파라미터

# train_loss = np.inf # train loss 초기값 무한대로 세팅
best_test_loss = np.inf

num_epochs = 1000

train_loss_list = []
test_loss_list = []
train_accuracy_list = []
test_accuracy_list = []

def get_accuracy_list(outputs, labels):
    outputs = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    estimates = np.argmax(outputs, 1)   # output.shape = (4,2)
    accuracy_list = []
    for estimate, label in zip(estimates, labels):
        if estimate==label:
            accuracy_list.append(1)
        else:
            accuracy_list.append(0)
    return accuracy_list                # accuracy_list = [1,1,0,1]

for epoch in range(num_epochs):
    print(f"Epoch: {epoch} / {num_epochs-1}")

    model.train()   # 학습 모드로 전환
    loss_sum = 0
    accuracy_list = []
    for images, labels in tqdm(train_loader, desc="Train"):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()     # weight 업데이트 할 값 계산 (편미분, Gradient)
        optimizer.step()    # weight 업데이트 실행
        loss_sum += loss.item()*batch_size
        accuracy_list += get_accuracy_list(outputs, labels)         # [1,1,0,1] + [0,0,1,1] + [0,1,0,1] = [1,1,0,1,0,0,1,1,0,1,0,1]
    train_loss = loss_sum/len(train_dataset)
    train_loss_list.append(train_loss)
    train_accuracy = np.mean(accuracy_list)*100 # %단위로 변환
    train_accuracy_list.append(train_accuracy)

    model.eval()    # 검증 모드로 전환
    with torch.no_grad():
        loss_sum = 0
        accuracy_list = []
        for images, labels in tqdm(test_loader, desc="Test"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item()*batch_size
            accuracy_list += get_accuracy_list(outputs, labels)
    test_loss = loss_sum/len(test_dataset)
    test_loss_list.append(test_loss)
    test_accuracy = np.mean(accuracy_list)*100
    test_accuracy_list.append(test_accuracy)

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        os.makedirs('output', exist_ok=True)    # 'output' 폴더가 존재하면 만들지 마라는 뜻.
        torch.save(model.state_dict(), os.path.join('output', 'model_best.pth'))

    print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}")
    print(f"Test Loss:  {test_loss:.4f}, Accuracy: {test_accuracy:.2f}")

    plt.figure(figsize=(5, 10))
    plt.subplot(2,1,1)
    plt.plot(train_loss_list, marker='o', label='Train Loss')
    plt.plot(test_loss_list, marker='o', label='Test Loss')
    plt.legend()
    plt.grid()
    plt.title("Loss Curve")

    plt.subplot(2,1,2)
    plt.plot(train_accuracy_list, marker='o', label='Train Accuracy')
    plt.plot(test_accuracy_list, marker='o', label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('%')
    plt.legend()
    plt.grid()
    plt.savefig("loss_curve.png")
    plt.close()

