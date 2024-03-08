import os
from medmnist import PathMNIST

# download 28x28 PathMNIST first, to see if it works, the do whit 224x224 since it is too big. 
def ensure_dir(directory):
   
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_medmnist(split, data_dir):
    ensure_dir(data_dir)
    
    try:
        dataset = PathMNIST(split=split, root=data_dir, download=True, size=28)
        print(f"Downloaded PathMNIST {split} data to {dataset.root}")
    except RuntimeError as e:
        print(f"Error downloading PathMNIST {split} data: {e}")

root_dir = os.path.join(os.getcwd(), 'data')

splits = ['train', 'val', 'test']
for split in splits:
    data_dir = os.path.join(root_dir, split)
    download_medmnist(split, data_dir)
