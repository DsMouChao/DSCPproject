import torch
import torch.nn as nn
import torch.nn.functional as F
from medmnist import PathMNIST
from torchvision import transforms
from torch.utils.data import DataLoader

# This is a most first model experiment for the PathMNIST dataset. 
# The algorithm for this model is very fast. 

# Define the LeNet model 
# Attention: This model is very simlpe with only 2 convolutional layers and 3 fully connected layers.
# The function of this model is to show how to use the PathMNIST dataset to train a simple model.
# The function of this model to set a baseline for the accuracy of the PathMNIST dataset.
# we should update the model or use another model to get a better accuracy.
class LeNet(nn.Module):
    def __init__(self, num_classes=9):# 9 classes
        super(LeNet, self).__init__()
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

transform = transforms.Compose([ # define the transformation
    transforms.ToTensor(),
    transforms.Normalize(mean =[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the PathMNIST dataset
train_set = PathMNIST(split='train', root='./data/train', transform=transform, download=True)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

# Load the validation set
val_set = PathMNIST(split='val', root='./data/val', transform=transform, download=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the LeNet model, loss function, and optimizer
model = LeNet(num_classes=9).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = torch.tensor([label[0] for label in labels]).to(device)

        optimizer.zero_grad() # zero the parameter gradients
        outputs = model(images) # forward pass
        loss = criterion(outputs, labels) # calculate the loss
        loss.backward() # backward pass
        optimizer.step() # update the weights

        running_loss += loss.item() # accumulate the loss
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")


model.eval()
total = 0
correct = 0
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = torch.tensor([label[0] for label in labels]).to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on the validation set: {100 * correct / total:.2f}%')
# After 10 epochs，the accuracy on the validation set is about 75%.
# The accuracy can be improved by using a deeper model, training for more epochs, or using a different optimizer.
#1. ResNet: Offers residual learning frameworks to ease the training of much deeper networks, well-suited for image classification tasks.
#2. DenseNet: Connects each layer to every other layer in a feed-forward fashion, well-suited for image classification tasks.
#3. VGG: A simple and widely used convolutional neural network, well-suited for image classification tasks.

#4. Rotating, translating, and scaling images. Applying random crops and flips to the images. Employing elastic deformations for non-linear transformations.
#5. Incorporating attention mechanisms, such as SENet (Squeeze-and-Excitation Networks), 
#    can help models focus on the most relevant parts of an image. 
#    This is particularly important in medical image processing, 
#    where specific regions often contain critical diagnostic information.

