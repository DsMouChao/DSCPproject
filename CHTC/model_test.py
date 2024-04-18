import argparse
import torch
import torch.utils.data as Data
from torchvision import transforms
from medmnist import PathMNIST
from model import LeNet, ResNet
import pandas as pd

def test_data_process(image_size):
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL images to tensors.
        transforms.Normalize((0.5,), (0.5,))  # Normalize images.
    ])
    
    test_data = PathMNIST(root='./data', split="test",size=image_size, transform=transform, download=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=2)
    return test_loader

def test_model_process(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    test_corrects = 0
    test_num = 0

    with torch.no_grad():
        for test_data_x, test_data_y in test_loader:
            test_data_x = test_data_x.to(device)
            test_data_y = torch.tensor([label[0] for label in test_data_y]).to(device)
            model.eval()

            test_output = model(test_data_x)
            test_preds = torch.argmax(test_output, 1)

            test_corrects += torch.sum(test_preds == test_data_y.data)
            test_num += test_data_y.size(0)

    test_acc = test_corrects.double() / test_num
    return test_acc.item()  # Return accuracy as a float

def load_model(model_type, image_size, use_se):
    if model_type == "LeNet":
        model = LeNet(num_classes=9, use_se=use_se,image_size=image_size)
    elif model_type == "ResNet":
        model = ResNet(num_classes=9, use_se=use_se,image_size=image_size)
    
    model_file = f"output_{model_type}_{image_size}x{image_size}{'_SE' if use_se else ''}.pth"
    state_dict = torch.load(model_file)
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    model.load_state_dict(filtered_state_dict, strict=False)
    return model

def main():
    parser = argparse.ArgumentParser(description="Test different models with different settings.")
    parser.add_argument("--model", type=str, required=True, help="Model type: LeNet or ResNet")
    parser.add_argument("--image_size", type=int, required=True, help="Input image size: 28 or 224")
    parser.add_argument("--use_se", action='store_true', help="Use SE blocks if specified")
    args = parser.parse_args()

    model = load_model(args.model, args.image_size, args.use_se)
    test_loader = test_data_process(args.image_size)
    test_acc = test_model_process(model, test_loader)
    unique_filename = 'test_results_{}_{}_{}.csv'.format(args.model, args.image_size, 'SE' if args.use_se else 'noSE')
    # Save results to CSV
    results = pd.DataFrame([{
        "Model": args.model,
        "Image_Size": args.image_size,
        "SE_Block": args.use_se,
        "Accuracy": test_acc
    }])
    results.to_csv(unique_filename, index=False)

if __name__ == '__main__':
    main()
