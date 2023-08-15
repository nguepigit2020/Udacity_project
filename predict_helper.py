import torch
from train_helper import MyModel 
from PIL import Image
from torchvision import datasets, transforms, models  



def process_image(image):
    
    # Define the transformations
    transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    
    # Apply the transformations to the image
    processed_image = transform(image)


    # Convert the PyTorch tensor to a NumPy array
    numpy_image = processed_image.numpy()
    # numpy_image = numpy_image.transpose(2, 0, 1)

    return numpy_image

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']
    model = MyModel(arch, hidden_units)
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model 


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    im = process_image(Image.open(image_path))
    im = torch.from_numpy(im)
    im.unsqueeze_(0)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ## Load the model and the image to device
    im = im.to(device, dtype=torch.float)
    model.to(device); 
    
    ## Make the prediction using our model
    ps = torch.exp(model(im)) 
    
    ##Looking for top probability and top class
    top_p, top_class = ps.topk(topk, dim=1)
    
    ##Convert the output of top probabilities to list
    top_p = top_p.cpu().detach().numpy()[0]
    
    ##Convert the output of top classes to list
    top_class = top_class.cpu().detach().numpy()[0]
    
    return top_p, top_class 
