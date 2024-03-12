import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F

# I make the model become a class, so that it can be used in other files.
# We could define every model in a single file, and then use them later in a efficient way.

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.sig =nn.Sigmoid()
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.f5 = nn.Linear(16*5*5, 120)
        self.f6 = nn.Linear(120, 84)
        self.f7 = nn.Linear(84, 10)

    
    def forward(self, x):
        x = self.c1(x)
        x = self.sig(x)
        x = self.s2(x)
        x = self.c3(x)
        x = self.sig(x)
        x = self.s4(x)
        x = self.flatten(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        return x

class LeNet2(nn.Module):
    def __init__(self, num_classes=9):# 9 classes
        super(LeNet2,self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)# 3 channels, 6 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, 5)# 6 input channels, 16 output channels, 5x5 kernel
        self.fc1 = nn.Linear(16*4*4, 120)# 16*4*4 input features, 120 output features
        self.fc2 = nn.Linear(120, 84)# 120 input features, 84 output features
        self.fc3 = nn.Linear(84, num_classes)  # 84 input features, 9 output features

    def forward(self, x):# x is the input image
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))# 2x2 max pooling
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)# 2x2 max pooling
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
    model = LeNet()
    print(summary(model, (1, 28, 28))) 
