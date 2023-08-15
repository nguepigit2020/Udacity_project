import os 
import numpy as np
import torch 
from torch import nn 
from torch import optim
import torch.nn.functional as F 
from torchvision import datasets, transforms, models   
from collections import OrderedDict 



def MyModel(arch, hidden_units):
        
    # The output of the classifier, this value depends on the problem.
    output_size = 102 
    
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
    elif arch == "resnet18":
        model = models.resnet18(pretrained=True)
    elif arch == "alexnet": 
        model = models.alexnet(pretrained=True)
    elif arch == "vgg19":
        model = models.vgg19(pretrained=True) 
    elif arch == "densenet161":
        model = models.densenet161(pretrained=True) 
    elif arch == "inception_v3": 
        model = models.inception_v3(pretrained=True) 
    else:
        model = models.vgg16(pretrained=True)
                   
    # Freeze parameters 
    for param in model.parameters():
        param.requires_grad = False

    
    ######### Getting the input size of the classifier ###############
    counter = 0 
    input_classifer = 0 

    for each_classifier in model.classifier.parameters():
        if counter == 1:
            break
        
        input_classifier = each_classifier.shape[1]
        counter += 1
    

    classifier = nn.Sequential(OrderedDict([
                                           ('fc1', nn.Linear(input_classifier, hidden_units)) , 
                                           ('relu1', nn.ReLU()), 
                                           ('dropout1', nn.Dropout(0.4)), 
                                           ('fc2', nn.Linear(hidden_units, output_size)),
                                           ('output', nn.LogSoftmax(dim=1))
                                           ]))

    model.classifier = classifier 
    
    return model 


def load_data(data_dir):
        
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'    
    
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_dataloaders = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloaders = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    valid_dataloaders = torch.utils.data.DataLoader(valid_dataset, batch_size=32)
            

    class_to_idx = train_dataset.class_to_idx

    
    return train_dataloaders, valid_dataloaders, test_dataloaders, class_to_idx 