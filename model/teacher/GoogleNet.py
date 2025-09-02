import torch
import torch.nn as nn


class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super().__init__()
        self.n1x1 = n1x1
        self.n3x3 = n3x3
        self.n5x5 = n5x5
        self.pool_planes = pool_planes

        # 1x1 conv branch
        if self.n1x1:
            self.branch1x1 = nn.Sequential(
                nn.Conv2d(in_planes, n1x1, kernel_size=1),
                nn.BatchNorm2d(n1x1),
                nn.ReLU(True),
            )

        # 1x1 conv -> 3x3 conv branch
        if self.n3x3:
            self.branch3x3 = nn.Sequential(
                nn.Conv2d(in_planes, n3x3red, kernel_size=1),
                nn.BatchNorm2d(n3x3red),
                nn.ReLU(True),
                nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
                nn.BatchNorm2d(n3x3),
                nn.ReLU(True),
            )

        # 1x1 conv -> 5x5 conv branch
        if self.n5x5 > 0:
            self.branch5x5 = nn.Sequential(
                nn.Conv2d(in_planes, n5x5red, kernel_size=1),
                nn.BatchNorm2d(n5x5red),
                nn.ReLU(True),
                nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
                nn.BatchNorm2d(n5x5),
                nn.ReLU(True),
                nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
                nn.BatchNorm2d(n5x5),
                nn.ReLU(True),
            )

        # 3x3 pool -> 1x1 conv branch
        if self.pool_planes > 0:
            self.branch_pool = nn.Sequential(
                nn.MaxPool2d(3, stride=1, padding=1),
                nn.Conv2d(in_planes, pool_planes, kernel_size=1),
                nn.BatchNorm2d(pool_planes),
                nn.ReLU(True),
            )

    def forward(self, x):
        out = []
        if self.n1x1:
            y1 = self.branch1x1(x)
            out.append(y1)
        if self.n3x3:
            y2 = self.branch3x3(x)
            out.append(y2)
        if self.n5x5:
            y3 = self.branch5x5(x)
            out.append(y3)
        if self.pool_planes:
            y4 = self.branch_pool(x)
            out.append(y4)
        return torch.cat(out, 1)


class GoogLeNet(nn.Module):
    def __init__(self, block=Inception, num_classes=1):
        super().__init__()
        
        # بخش اول: جایگزینی pre_layers با لایه‌های مشخص
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # بخش دوم: تغییر نام inception_a3 به inception3a و غیره
        self.inception3a = block(64, 64, 96, 128, 16, 32, 32)
        self.inception3b = block(256, 128, 128, 192, 32, 96, 64)
        
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception4a = block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = block(528, 256, 160, 320, 32, 128, 128)
        
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception5a = block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = block(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # بخش سوم: تغییر نام linear به fc
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool1(out)
        
        out = self.inception3a(out)
        out = self.inception3b(out)
        out = self.maxpool2(out)
        
        out = self.inception4a(out)
        out = self.inception4b(out)
        out = self.inception4c(out)
        out = self.inception4d(out)
        out = self.inception4e(out)
        out = self.maxpool3(out)

        out = self.inception5a(out)
        out = self.inception5b(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out, feature_list

def GoogLeNet_deepfake():
    return GoogLeNet(block=Inception, num_classes=1)
