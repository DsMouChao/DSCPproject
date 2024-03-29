import copy
import time
import pandas as pd
from medmnist import PathMNIST
from torchvision import transforms
import numpy as np
import torch
import matplotlib.pyplot as plt
from model import LeNet,LeNet2

# This file as template for the training process of the model, 
# no need to modify this file bigly for later use.
# This file could save the best model parameters and generate the training process graph.

def train_val_data_process():
    # Load the training and validation data
    train_data = PathMNIST(root='./data/train', split="train",transform= transforms.Compose([transforms.Resize(28),transforms.ToTensor()]), download=False)

    train_data, val_data = torch.utils.data.random_split(train_data, [round(0.8*len(train_data)), round(0.2*len(train_data))])
    
    # Create the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=8)

    val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=True, num_workers=8)
    
    # Return the dataloaders
    return train_loader, val_loader


def train_model_process(model,train_loader,val_loader,num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# Check if a GPU is available

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Define the optimizer.

    criterion = torch.nn.CrossEntropyLoss() # Define the loss function, 
    
    model.to(device) # Move the model to the GPU if available

    best_model_wts = copy.deepcopy(model.state_dict()) # copy the model parameters, to store the best model parameters
    
    # Initialize the parameters
    best_acc = 0.0 # initialize the best accuracy
    train_loss_all =[] # initialize the training loss, used to record the training loss of each epoch
    val_loss_all = [] # initialize the validation loss, used to record the validation loss of each epoch
    train_acc_all = [] # initialize the training accuracy, used to record the training accuracy of each epoch
    val_acc_all = [] # initialize the validation accuracy, used to record the validation accuracy of each epoch
    
    since = time.time() # Record the start time of training

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        train_loss = 0.0 # record the training loss of current epoch
        train_corrects = 0 # record the number of correct predictions of current epoch
        val_loss =0.0 # record the validation loss of current epoch
        val_corrects = 0 # record the number of correct predictions of current epoch
        train_num = 0 # record the number of training samples of current epoch
        val_num = 0 # record the number of validation samples of current epoch

        for step, (b_x, b_y) in enumerate(train_loader):
            # Move the data to the GPU if available
            b_x = b_x.to(device)
            # Move the data to the GPU if available
            b_y = torch.tensor([label[0] for label in b_y]).to(device)

            model.train()  # Set model to training mode

            outputs = model(b_x)  # Forward pass: compute predicted y by passing x to the model

            pre_lab = torch.argmax(outputs, 1)  # Get the predicted labels, return the index of the maximum value

            loss = criterion(outputs, b_y) 

            optimizer.zero_grad()  # Clear the gradients of all optimized variables
            # backward + optimize only if in training phase
            loss.backward()

            optimizer.step()  # Update the model parameters

            train_loss += loss.item() * b_x.size(0)
            train_corrects += torch.sum(pre_lab == b_y.data)
            train_num += b_x.size(0)

        for step, (b_x, b_y) in enumerate(val_loader):
            # Move the data to the GPU if available
            b_x = b_x.to(device)
            # Move the data to the GPU if available
            b_y = torch.tensor([label[0] for label in b_y]).to(device)
            
            # Set model to training mode
            model.eval() #no need to calculate the gradients
            outputs = model(b_x)
            pre_lab = torch.argmax(outputs, 1)
            loss = criterion(outputs, b_y)

            val_loss += loss.item() * b_x.size(0)
            val_corrects += torch.sum(pre_lab == b_y.data)
            val_num += b_x.size(0)
        
        # store the loss and accuracy of each epoch
        train_loss_all.append(train_loss/train_num)
        val_loss_all.append(val_loss/val_num)
        train_acc_all.append(train_corrects.double()/train_num)
        val_acc_all.append(val_corrects.double()/val_num)

        print('Epoch: {} Train Loss: {:.4f} Acc: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print('Epoch: {} Val Loss: {:.4f} Acc: {:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))

        # store the best model parameters
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        
        # save the best model parameters
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
    # save the best model parameters
    torch.save(best_model_wts, 'best_model.pth')

    train_process = pd.DataFrame(data={"epoch":range(num_epochs),'train_loss':train_loss_all,'val_loss':val_loss_all,'train_acc':train_acc_all,'val_acc':val_acc_all})
    return train_process
    
def graph(train_procss):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_procss['epoch'],train_process["train_loss"],"ro-",label='train_loss')
    plt.plot(train_procss['epoch'],train_process["val_loss"],"bs-",label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
   

    plt.subplot(1, 2, 2)
    plt.plot(train_procss['epoch'],train_process["train_acc"],"ro-",label='train_acc' )
    plt.plot(train_procss['epoch'],train_process["val_acc"],"bs-",label='val_acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.show()

if __name__ == "__main__":
    lenet = LeNet2(9)
    train_dataloader, val_dataloader = train_val_data_process()
    train_process = train_model_process(lenet,train_dataloader,val_dataloader,10)
    graph(train_process)




   



        

        


















             




    
    
    















