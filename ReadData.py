
import numpy as np
import matplotlib.pyplot as plt
from medmnist import PathMNIST
from torchvision.transforms.functional import to_pil_image, to_tensor 
from torchvision import transforms

# Load the PathMNIST dataset
data_dir = 'data/train' 
dataset = PathMNIST(split='train', root=data_dir, download=False)

# Display the first 5 images and their labels
fig, axes = plt.subplots(1, 5, figsize=(15, 3)) # 1 row, 5 columns
for i, ax in enumerate(axes):
    # Get the i-th image and its label
    img, label = dataset[i]  # img is a PIL image, label is an integer

    # Display the image and label
    ax.imshow(np.array(img), cmap='gray')  # Convert the PIL image to a numpy array
    ax.set_title(f'Label: {label}')
    ax.axis('off')  # Hide the axes
plt.show()


# Define the transformation - only ToTensor() is needed to convert PIL images to PyTorch tensors
transform = transforms.Compose([
    transforms.ToTensor()
])
# Loading the dataset with transformations
data_dir = './data/train'
dataset = PathMNIST(split='train', root=data_dir, transform=transform, download=False)


# Fetch the first image and its label from the dataset
img_tensor, label = dataset[0]  # img_tensor is a PyTorch tensor, label is an integer

# Convert the image tensor to a NumPy array
img_numpy = img_tensor.numpy()
img_numpy = np.squeeze(img_numpy)  # Remove the channel dimension if it's 1

# Print the NumPy array of the image and the label
print("NumPy array of the first image:")
print(img_numpy)
print(f"Shape of the array: {img_numpy.shape}")
print(f"Label of the first image: {label}")


