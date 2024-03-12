import torch
import torch.utils.data as Data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from medmnist import PathMNIST
from model import LeNet,LeNet2

# This file is as a template for the testing process of the model,
# No need to modify this file bigly for later use.

def test_data_process():
    # Load the training and validation data
    test_data = PathMNIST(root='./data/train', split="test",transform= transforms.Compose([transforms.Resize(28),transforms.ToTensor()]), download=False)


    # Create the dataloaders
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=2)

    
    # Return the dataloaders
    return test_loader

def test_model_process(model,test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# Check if a GPU is available

    model = model.to(device) # Move the model to the GPU if available，

    test_corrects = 0 #record the number of correct predictions
    test_num = 0 #record the number of test samples
    model.eval() # Set the model to evaluation mode

    # Iterate over the test data
    with torch.no_grad():
        for test_data_x, test_data_y in test_loader:
            test_data_x = test_data_x.to(device)
            test_data_y = torch.tensor([label[0] for label in test_data_y]).to(device)

            model.eval()  # Set model to evaluation mode

            test_output = model(test_data_x)
            test_preds = torch.argmax(test_output, 1)

            test_corrects += torch.sum(test_preds == test_data_y.data)
            test_num += test_data_y.size(0)

    test_acc = test_corrects.double() / test_num
    print('Test Acc: {:.4f}'.format(test_acc))

if __name__ == '__main__':
    model = LeNet2(9)
    model.load_state_dict(torch.load('best_model.pth'))
    test_loader = test_data_process()
    test_model_process(model,test_loader)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# Check if a GPU is available
    # model = model.to(device) # Move the model to the GPU if available
    # class_names = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']
    # with torch.no_grad():
    #     for test_data_x, test_data_y in test_loader:
    #         test_data_x = test_data_x.to(device)
    #         test_data_y = test_data_y.to(device)

    #         model.eval()
    #         test_output = model(test_data_x)
    #         test_preds = torch.argmax(test_output, 1)
    #         result = test_preds.item()
    #         label = test_data_y.item()

    #         print('Predicted:', class_names[result]," --- " 'Label:', class_names[label])





