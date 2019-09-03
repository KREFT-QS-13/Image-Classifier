import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F

from get_extra_input_train import  get_extra_input

arg = get_extra_input()
        
# editing data
data_dir = arg.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

train_transforms = transforms.Compose([ transforms.RandomVerticalFlip(),
                                        transforms.RandomRotation(0),
                                        transforms.RandomResizedCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ])

val_transforms = transforms.Compose([ transforms.RandomResizedCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ])

train_datasets =  datasets.ImageFolder(train_dir, transform=train_transforms)
val_datasets = datasets.ImageFolder(valid_dir, transform=val_transforms)
      
trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
valloaders = torch.utils.data.DataLoader(val_datasets, batch_size=64, shuffle=True)
        
# Setting the model
if arg.arch == "alexnet":
    model = models.alexnet(pretrained = True)
    input_layer = 9216 
else:
    model = models.vgg16_bn(pretrained = True)
    input_layer = 25088
            
for p in model.parameters():
    p.requires_grad = False 
        
classifier = nn.Sequential(nn.Linear(input_layer, arg.hidden_units),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(arg.hidden_units, arg.hidden_units // 2),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(arg.hidden_units // 2, 102),
                           nn.LogSoftmax(dim=1))  

model.classifier = classifier
device = torch.device("cuda" if arg.gpu else "cpu")
model.to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=arg.learning_rate)
epochs = arg.epochs
       
#training part
print("The model is learning from data...", "It can take a while...")
for e in range(epochs):
    print("Epoch: {}/{}: ".format(e+1, epochs))
    running_loss = 0
    for inputs, labels in trainloaders:
        inputs, labels = inputs.to(device), labels.to(device)
                
        optimizer.zero_grad()
               
        log_ps = model.forward(inputs)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
               
        running_loss += loss.item()
    else:
        test_loss = 0
        test_accuracy = 0
        with torch.no_grad():
            model.eval()
            for images, labels in valloaders:
                images, labels = images.to(device), labels.to(device)
                        
                log_ps = model.forward(images)
                test_loss += criterion(log_ps, labels)
                        
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        model.train()
                
        print("Trainig Loss: {:.3f}..".format(running_loss/len(trainloaders)),
              "Test Loss: {:.3f}..".format(test_loss/len(valloaders)),
              "Test Accuracy: {:.2f}%".format( (test_accuracy/len(valloaders)) * 100 ))
        
#       saving result
checkpoint = { 'classifier': model.classifier,
               'arch': arg.arch,
               'epochs': arg.epochs,
               'state_dict': model.state_dict(),
               'class_to_idx':  train_datasets.class_to_idx }
torch.save(checkpoint, arg.save_dir + '/checkpoint.pth')     
print("Progress has been saved in file, here: {}".format(arg.save_dir + '/checkpoint.pth'))