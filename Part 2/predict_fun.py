import torch
from torchvision import datasets, transforms, models
from process_image import process_image

def predict(image_path, model, top_k, device):
    im = process_image(image_path)
    im = im.unsqueeze_(0)
    device = torch.device("cuda" if device else "cpu")

    with torch.no_grad():
        model.eval()
        im = im.to(device)
        log_ps = model.forward(im)
        ps = torch.exp(log_ps)
    
        probs, classes = ps.topk(top_k)
    
    if(device):
        probs = probs.cpu().detach().numpy().tolist()[0]
        classes = classes.cpu().detach().numpy().tolist()[0]
    else:   
        probs = probs.detach().numpy().tolist()[0]
        classes = classes.detach().numpy().tolist()[0]
    
    idx_to_class = {val: x for x, val in model.class_to_idx.items()}
    
    labels = [idx_to_class[x] for x in classes]
    
    return probs, labels