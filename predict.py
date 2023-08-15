import argparse
from PIL import Image
from predict_helper import load_checkpoint
from predict_helper import predict 
import json
import sys 
import torch

parser = argparse.ArgumentParser(description='Predicting a flower name from an image')

parser.add_argument('path_to_image', action="store")
parser.add_argument('checkpoint', action="store")
parser.add_argument('--top_k', action="store",type=int, dest="top_k", default=5)
parser.add_argument('--category_names', action="store", dest="category_names", default="")
parser.add_argument('--gpu', action="store_true", dest="gpu", default=False)


results = parser.parse_args()

path_to_image = results.path_to_image
checkpoint = results.checkpoint
top_k = results.top_k
category_names = results.category_names
gpu = results.gpu


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
model = load_checkpoint(checkpoint)

model.to(device); 

class_to_idx = model.class_to_idx 

probs, classes = predict(path_to_image, model, top_k) 

if not category_names:
    print(classes) 
    print(probs) 
    
else:   
    name_classes = []
    
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)    
    
    for i in classes:
        for clas,index in class_to_idx.items():
            if index == i:
                flower_key = clas
        name_classes.append(cat_to_name[flower_key])
    
    print(name_classes)
    print(probs)