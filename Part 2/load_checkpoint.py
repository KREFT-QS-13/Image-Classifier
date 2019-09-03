import torch
from torchvision import models


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    model = models.vgg16_bn(pretrained=True) if checkpoint['arch']=='vgg' else models.alexnet(pretrained=True)
    model.classifier = checkpoint['classifier'] 
    model.load_state_dict(checkpoint['state_dict'])
    epochs = checkpoint['epochs']
    model.class_to_idx = checkpoint['class_to_idx']
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model