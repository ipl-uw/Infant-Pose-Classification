import torch
import torch.nn as nn


class TNet(nn.Module):
    def __init__(self):
        super(TNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 3), stride=(1, 1)),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=(13, 1), stride=(1, 1)),
            nn.BatchNorm2d(512),
        )
        self.fully_connected = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, 9)
        )

    def forward(self, x):
        temp = x.unsqueeze(1)
        temp = self.conv(temp)
        temp.squeeze_()
        temp = self.fully_connected(temp)
        temp = temp.view((temp.shape[0], 3, 3))
        return temp

class FeatureTNet(nn.Module):
    def __init__(self):
        super(FeatureTNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=(13, 1), stride=(1, 1)),
            nn.BatchNorm2d(512),
        )
        self.fully_connected = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64* 64)
        )

    def forward(self, x):
        temp = self.conv(x)
        temp.squeeze_()
        temp = self.fully_connected(temp)
        temp = temp.view((temp.shape[0], 64, 64))
        return temp


if __name__ == '__main__':
    a = TNet()
    b = torch.zeros((2, 13, 3))
    print(a(b).shape)
