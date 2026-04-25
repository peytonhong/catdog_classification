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
from PIL import Image

# transform = transforms.Compose([transforms.ToTensor()])
transform = transforms.Compose([
    transforms.Resize((128, 128)), # 크기 고정 확인
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = CatDogClassification(dataset_dir=os.path.join('catdog_dataset', 'train'), transforms=transform)
test_dataset = CatDogClassification(dataset_dir=os.path.join('catdog_dataset', 'test'), transforms=transform)

# image, label = train_dataset[10]
batch_size = 64
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

# images, labels = next(iter(train_loader))
# images = images.numpy()
# cv2.imshow('image', images[0])
# print(images.shape)
# print(labels)
# cv2.waitKey()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = MLPModel()
model = CNNModel_v2().to(device)

# criterion = nn.MSELoss(reduction='mean')
criterion = nn.CrossEntropyLoss()

# optimizer = optim.SGD(model.parameters(), lr=0.001) # lr: learning rate
optimizer = optim.Adam(model.parameters(), lr=0.0001)    # Adam: learning rate를 자동 튜닝해줌
# Hyperparmeter Tuning: 모델 구조, lr 값, optimzer 종류 등 학습에 영향을 미치는 파라미터

# train_loss = np.inf # train loss 초기값 무한대로 세팅
best_test_loss = np.inf

num_epochs = 100

train_loss_list = []
test_loss_list = []

# load model
state_dict = torch.load(os.path.join('output', 'model_best.pth'), weights_only=True, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)



model.eval()    # 검증 모드로 전환
with torch.no_grad():
    # for images, labels in tqdm(test_loader, desc="Test"):
    image_paths = glob(os.path.join("catdog_dataset", "test", "**", "*.jpg"))
    np.random.shuffle(image_paths)
    idx = 0
    while True:
        image_path = image_paths[idx]
        image_cv = cv2.imread(image_path)
        image = Image.open(image_paths[idx]).convert('RGB')
        # temporary comment 
        outputs = model(transform(image).unsqueeze(0).to(device))   # input shape: (C, H, W) --> (1, C, H, W)
        outputs = outputs[0].detach().cpu().numpy()
        label = np.argmax(outputs)
        print(outputs, label)
        cv2.imshow("image", image_cv)
        key = cv2.waitKey()

        if key==ord('q'):
            break
        elif key==ord('a'):
            idx -= 1
        elif key==ord('d'):
            idx += 1
        if idx < 0:
            idx = 0
        if idx > len(image_paths)-1:
            idx = len(image_paths)
