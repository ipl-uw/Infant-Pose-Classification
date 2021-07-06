import torch
import torch.nn as nn
from tnet import TNet, FeatureTNet
from torch.nn import functional as F

class PoseNet(nn.Module):
    def __init__(self):
        super(PoseNet, self).__init__()
        self.tnet = TNet()
        # self.ftnet = FeatureTNet()
        # self.conv0 = nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size=(1, 3), stride=(1, 1)),
        #     nn.PReLU(),
        #     nn.BatchNorm2d(64),
        #     nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1)),
        # )
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 3), stride=(1, 1)),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1)),
            nn.PReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 1024, kernel_size=(13, 1), stride=(1, 1)),
            nn.BatchNorm2d(1024),
        )
        self.fully_connected = nn.Sequential(
            nn.Linear(1024, 512),
            nn.PReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
        )
        self.direct_output = nn.Linear(256, 11)

    def forward(self, x,y):
        transform = self.tnet(x)
        x = torch.bmm(x, transform)
        x.unsqueeze_(1)
        # x = self.conv0(x)
        # transform = self.ftnet(x)
        # x.transpose_(2, 1)
        # x = torch.matmul(x, transform)
        # x.transpose_(2, 1)
        x = self.conv(x)
        x.squeeze_()
        x = self.fully_connected(x)        
        x_direct = self.direct_output(x)

        x_direct = x_direct*y
        
        return x_direct


if __name__ == '__main__':
    test = torch.zeros((2, 13, 3))
    p = PoseNet()
    p(test)
