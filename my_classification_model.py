import torch.nn as nn
import numpy as np
import torch
import cv2
import os

class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(16384, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 2)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.sigmoid(x)

        x = self.fc3(x)
        # x = self.sigmoid(x)
        x = self.softmax(x)
        return x

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)    
        self.maxpool = nn.MaxPool2d(2, 2)               

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)    
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)

        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(32768, 1000)
        self.fc2 = nn.Linear(1000, 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()                                                        

    def forward(self, x):       # 128x128x3
        x = self.conv1(x)       # 128x128x32
        x = self.relu(x)
        x = self.conv2(x)       # 128x128x32
        x = self.relu(x)
        x = self.maxpool(x)     # 64x64x32

        x = self.conv3(x)       # 64x64x64
        x = self.relu(x)
        x = self.conv4(x)       # 64x64x64
        x = self.relu(x)
        x = self.maxpool(x)     # 32x32x64

        x = self.conv5(x)       # 32x32x128
        x = self.relu(x)
        x = self.conv6(x)       # 32x32x128
        x = self.relu(x)
        x = self.maxpool(x)     # 16x16x128

        x = self.flatten(x)     # 32768
        x = self.fc1(x)         # 1000
        x = self.relu(x)        # 1000

        x = self.fc2(x)         # 2
        # x = self.sigmoid(x)     # 2

        return x

class CNNModel_v2(nn.Module):
    def __init__(self):
        super(CNNModel_v2, self).__init__()
        
        # 반복되는 구조를 정의하기 편하게 구성
        def conv_block(in_f, out_f):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, 3, padding=1),
                nn.BatchNorm2d(out_f), # BN 추가
                nn.ReLU()
            )

        self.layer1 = conv_block(3, 32)
        self.layer2 = conv_block(32, 32)
        
        self.layer3 = conv_block(32, 64)
        self.layer4 = conv_block(64, 64)

        self.layer5 = conv_block(64, 128)
        self.layer6 = conv_block(128, 128)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        
        # FC 레이어 크기 최적화 및 Dropout 추가
        self.fc1 = nn.Linear(16 * 16 * 128, 512) 
        self.dropout = nn.Dropout(0.5) # 과적합 방지
        self.fc2 = nn.Linear(512, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer2(self.layer1(x))
        x = self.maxpool(x) # 64x64

        x = self.layer4(self.layer3(x))
        x = self.maxpool(x) # 32x32

        x = self.layer6(self.layer5(x))
        x = self.maxpool(x) # 16x16

        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x




# model = MLPModel()

# # input_data = torch.rand(784)
# # # print(input_data)
# # output_data = model(input_data)
# # print(output_data)

# image = cv2.imread(os.path.join('dataset', 'cat', '0.jpg'), cv2.IMREAD_GRAYSCALE)
# image = cv2.resize(image, (28,28))
# image = torch.tensor(image, dtype=torch.float32)
# output_data = model(image)
# print(output_data)
