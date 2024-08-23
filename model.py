import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torch
import matplotlib.pyplot as plt
import numpy as np

def get_model(num_classes):
    model = maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

def train_model(model, data_loader, device, num_epochs):
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        lr_scheduler.step()
        print(f"Epoch #{epoch} loss: {losses.item()}")

    return model

def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            outputs = model(images)
            all_preds.extend(outputs)
            all_targets.extend(targets)
    return all_preds, all_targets

def visualize_predictions(model, dataset, device, num_images=5):
    model.eval()
    fig, axs = plt.subplots(num_images, 2, figsize=(10, num_images * 5))
    with torch.no_grad():
        for i in range(num_images):
            img, target = dataset[i]
            img = img.to(device)
            output = model([img])[0]
            img = img.permute(1, 2, 0).cpu().numpy()
            axs[i, 0].imshow(img, cmap='gray')
            axs[i, 0].set_title("Original Image")

            masks = output['masks'].cpu().numpy()
            combined_mask = np.zeros_like(masks[0][0])
            for mask in masks:
                combined_mask = np.maximum(combined_mask, mask[0])
            axs[i, 1].imshow(combined_mask, cmap='gray')
            axs[i, 1].set_title("Predicted Mask")
    plt.show()
