import torch
from torch.utils.data import DataLoader
from nuclei_dataset import NucleiDataset, collate_fn
from model import get_model
from model import evaluate_model, visualize_predictions

def main():
    val_image_dir = "/Users/crispypig/Desktop/val/images"
    val_mask_dir = "/Users/crispypig/Desktop/val/masks"
    num_classes = 2  # Background and nuclei
    batch_size = 2

    # Validation dataset and DataLoader
    val_dataset = NucleiDataset(val_image_dir, val_mask_dir)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = get_model(num_classes)
    model.load_state_dict(torch.load("nuclei_segmentation_model.pth"))
    model.to(device)

    # Evaluation
    preds, targets = evaluate_model(model, val_data_loader, device)
    print(preds, targets)

    # Visualize predictions
    visualize_predictions(model, val_dataset, device)

if __name__ == "__main__":
    main()
