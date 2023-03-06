import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 =nn.Conv2d(3,6,5)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1= nn.Linear(16*5*5,120)
        self.fc2= nn.Linear(120,84)
        self.fc3= nn.Linear(84,10)
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class InceptionModule(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(InceptionModule,self).__init__()
        self.branch1=nn.Sequential(nn.Conv2d(in_channels,out_channels=out_channels,kernel_size=1,stride=1,padding=0),nn.ReLU())
        conv3_1 = nn.Conv2d(in_channels,out_channels=out_channels,kernel_size=1,stride=1,padding=0)
        conv3_3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.branch2 = nn.Sequential(conv3_1, conv3_3,nn.ReLU())

        conv5_1 = nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        conv5_5 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.branch3 = nn.Sequential(conv5_1,conv5_5,nn.ReLU())

        max_pool_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        conv_max_1 = nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.branch4 = nn.Sequential(max_pool_1, conv_max_1,nn.ReLU())

    def forward(self, input):
        output1 = self.branch1(input)
        output2 = self.branch2(input)
        output3 = self.branch3(input)
        output4 = self.branch4(input)
        return torch.cat([output1, output2, output3, output4], dim=1)

class Autoencoder_linear(nn.Module):
    def __init__(self):
        super(Autoencoder_linear,self).__init__()

        #encoder
        self.enc1 = nn.Linear(in_features=784,out_features=256)
        self.enc2 = nn.Linear(in_features=256,out_features=128)
        self.enc3 = nn.Linear(in_features=128,out_features=64)
        self.enc4 = nn.Linear(in_features=64,out_features=32)
        self.enc5 = nn.Linear(in_features=32,out_features=16)

        #decoder
        self.dec1 = nn.Linear(in_features=16,out_features=32)
        self.dec2 = nn.Linear(in_features=32,out_features=64)
        self.dec3 = nn.Linear(in_features=64,out_features=128)
        self.dec4 = nn.Linear(in_features=128,out_features=256)
        self.dec5 = nn.Linear(in_features=256,out_features=784)

    def forward(self,x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x))

        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.relu(self.dec5(x))
        return x
