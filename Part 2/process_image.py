import torch
from torchvision import datasets, transforms, models
from PIL import Image

def process_image(image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    im = Image.open(image)
    trans = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
   
    return trans(im)