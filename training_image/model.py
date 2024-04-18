import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F

# I make the model become a class, so that it can be used in other files.
# We could define every model in a single file, and then use them later in a efficient way.

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels,bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, use_se=False ,stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        if use_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.se:
            y = self.se(y)
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.relu(y)
    
class ResNet(nn.Module):
    def __init__(self, num_classes=9, use_se=False,image_size=28):
        super(ResNet, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
        self.b2 = nn.Sequential(
            Residual(64, 64, use_se=use_se),
            Residual(64, 64, use_se=use_se)
        )
        self.b3 = nn.Sequential(
            Residual(64, 128, use_1x1conv=True, use_se = use_se, stride=2),
            Residual(128, 128, use_se=use_se)
        )
        self.b4 = nn.Sequential(
            Residual(128, 256, use_1x1conv=True, use_se=use_se, stride=2),
            Residual(256, 256, use_se=use_se)
        )
        self.b5 = nn.Sequential(
            Residual(256, 512, use_1x1conv=True, use_se=use_se, stride=2),
            Residual(512, 512, use_se=use_se)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class AlexNet(nn.Module):
    def __init__(self, num_classes=9, use_se=False,image_size=28):
        super(AlexNet, self).__init__()
        if image_size ==28:
            final_conv_size = 9216
        elif image_size == 224:
            final_conv_size = 774400

        self.ReLU = nn.ReLU()
        self.c1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.s2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.se1 = SEBlock(96) if use_se else nn.Identity()
        self.c3 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.s4 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.se2 = SEBlock(256) if use_se else nn.Identity()
        self.c5 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.se3 = SEBlock(384) if use_se else nn.Identity()
        self.c6 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.se4 = SEBlock(384) if use_se else nn.Identity()
        self.c7 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.se5 = SEBlock(256) if use_se else nn.Identity()
        self.flatten = nn.Flatten()
        self.f9 = nn.Linear(final_conv_size, 4096)
        self.f10 = nn.Linear(4096, 4096)
        self.f11 = nn.Linear(4096, num_classes)
    
    def forward(self, x):
        x = self.ReLU(self.c1(x))
        x = self.s2(x)
        x = self.se1(x)
        x = self.ReLU(self.c3(x))
        x = self.s4(x)
        x = self.se2(x)
        x = self.ReLU(self.c5(x))
        x = self.se3(x)
        x = self.ReLU(self.c6(x))
        x = self.se4(x)
        x = self.ReLU(self.c7(x))
        x = self.se5(x)

        x = self.flatten(x)
        x = self.ReLU(self.f9(x))
        x = F.dropout(x, 0.5)
        x = self.ReLU(self.f10(x))
        x = F.dropout(x, 0.5)
        x = self.f11(x)
        return x
    

class LeNet(nn.Module):
    def __init__(self, num_classes=9,use_se=False,image_size=28):# 9 classes
        super(LeNet,self).__init__()
        if image_size == 28:
            final_linear_size = 16 * 4 * 4
        elif image_size == 224:
            final_linear_size = 16 * 53 * 53

        self.conv1 = nn.Conv2d(3, 6, 5)# 3 channels, 6 output channels, 5x5 kernel
        self.se1 = SEBlock(6) if use_se else nn.Identity()
        self.conv2 = nn.Conv2d(6, 16, 5)# 6 input channels, 16 output channels, 5x5 kernel
        self.se2 = SEBlock(16) if use_se else nn.Identity()
        self.fc1 = nn.Linear(final_linear_size, 120)# 16*4*4 input features, 120 output features
        self.fc2 = nn.Linear(120, 84)# 120 input features, 84 output features
        self.fc3 = nn.Linear(84, num_classes)  # 84 input features, 9 output features

    def forward(self, x):# x is the input image
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))# 2x2 max pooling
        x = self.se1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)# 2x2 max pooling
        x = self.se2(x)
        x = x.view(-1, self.num_flat_features(x))# flatten the tensor
        x = F.relu(self.fc1(x))# ReLU activation
        x = F.relu(self.fc2(x))# ReLU activation
        x = self.fc3(x)# output layer
        return x
    
    def num_flat_features(self, x):# calculate the number of features
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1  # initialize the number of features
        for s in size: # multiply the dimensions
            num_features *= s # multiply the dimensions
        return num_features # return the number of features
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet(9,use_se=True).to(device)
    print(summary(model, (3, 28, 28))) 
