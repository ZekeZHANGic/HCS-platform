import torch
from torch.utils.data import DataLoader
from nuclei_dataset import NucleiDataset, collate_fn
from model import get_model, train_model
from inference import evaluate_model, visualize_predictions

def main():
    image_dir = "/Users/crispypig/Desktop/raw"
    mask_dir = "/Users/crispypig/Desktop/masks"
    num_classes = 2  # Background and nuclei
    num_epochs = 20
    batch_size = 5

# Training dataset and DataLoader
    train_dataset = NucleiDataset(image_dir, mask_dir)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)

    #device = torch.device("mps")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = get_model(num_classes)
    model = train_model(model, train_data_loader, device, num_epochs)

    torch.save(model.state_dict(), "nuclei_segmentation_model.pth")

if __name__ == "__main__":
    main()